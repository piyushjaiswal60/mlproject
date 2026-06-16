# ============================================================
# FILE: inference_pipeline.py
# ============================================================
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, correlate
from scipy.ndimage import gaussian_filter1d
import os
import glob

from plug.unet_model import get_model

# ============================================================
# CONFIG
# ============================================================
PIPE_DIAMETER_MM = 50.0
BEAD_DIAMETER_MM = 3.0
EFFECTIVE_FPS    = 6.0
FULL_PLUG_RATIO  = 0.92
MODEL_PATH       = 'checkpoints/best_model.pth'
IMG_SIZE         = (256, 768)   # must match training


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path, device):
    ckpt  = torch.load(model_path,
                        map_location=device)
    model = get_model(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded from {model_path}")
    print(f"  Trained val IoU: "
          f"{ckpt.get('val_iou', 'N/A'):.4f}")
    return model


# ============================================================
# STEP 2: U-NET SEGMENTATION
# ============================================================

def segment_with_unet(frame, model, device,
                       img_size=IMG_SIZE,
                       threshold=0.5,
                       debug_folder=None,
                       frame_idx=0,
                       save_debug=False):
    """
    Run U-Net on a single frame.
    Returns:
        plug_mask  : (H_orig, W_orig) uint8 binary mask
        prob_map   : (H_orig, W_orig) float32 probability
        smooth_top : (W_orig,) int array of plug top per column
    """
    gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(frame.shape)==3 else frame.copy())
    H_orig, W_orig = gray.shape

    # Resize to model input size
    resized = cv2.resize(gray,
                          (img_size[1], img_size[0]))

    # Normalize [0,1] and to tensor
    inp = resized.astype(np.float32) / 255.0
    inp = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0)
    inp = inp.to(device)

    # Forward pass
    with torch.no_grad():
        prob = model(inp)                   # (1,1,H,W)

    prob_np = prob.squeeze().cpu().numpy()  # (H,W) in model size

    # Resize probability map back to original
    prob_full = cv2.resize(prob_np,
                            (W_orig, H_orig),
                            interpolation=cv2.INTER_LINEAR)

    # Threshold → binary mask
    plug_mask = (prob_full > threshold).astype(np.uint8)*255

    # Post-process: close small holes, remove noise
    k1 = np.ones((9,9), np.uint8)
    k2 = np.ones((5,5), np.uint8)
    plug_mask = cv2.morphologyEx(plug_mask,
                                  cv2.MORPH_CLOSE, k1)
    plug_mask = cv2.morphologyEx(plug_mask,
                                  cv2.MORPH_OPEN,  k2)

    # Column-fill from bottom (plug is contiguous at bottom)
    plug_mask = enforce_bottom_connected(plug_mask, H_orig)

    # Extract smooth top boundary
    smooth_top = get_smooth_top_boundary(
        plug_mask, H_orig, W_orig
    )

    # ── Debug ────────────────────────────────────────────────
    if debug_folder and save_debug:
        sf = os.path.join(debug_folder,
                f'step2_unet/frame_{frame_idx:04d}')
        os.makedirs(sf, exist_ok=True)

        # A: original
        _save_img(gray, sf, 'A_original.png',
                  'A: Input frame')

        # B: probability map (heatmap)
        prob_vis = cv2.applyColorMap(
            (prob_full*255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        _save_img(prob_vis, sf,
                  'B_probability_map.png',
                  'B: U-Net probability (red=plug)')

        # C: raw binary mask
        _save_img(plug_mask, sf,
                  'C_binary_mask.png',
                  f'C: Binary mask (thresh={threshold})')

        # D: overlay with smooth boundary
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        ov  = vis.copy()
        ov[plug_mask>0] = (0,160,0)
        vis = cv2.addWeighted(vis, 0.5, ov, 0.5, 0)
        for x in range(W_orig-1):
            y1 = int(np.clip(smooth_top[x],  0,H_orig-1))
            y2 = int(np.clip(smooth_top[x+1],0,H_orig-1))
            cv2.line(vis,(x,y1),(x+1,y2),(0,0,255),2)
        _save_img(vis, sf, 'D_overlay.png',
                  'D: Overlay (green=plug, red=boundary)')

        # E: probability histogram
        fig, ax = plt.subplots(figsize=(6,3))
        ax.hist(prob_full.ravel(), bins=64,
                color='steelblue', alpha=0.8)
        ax.axvline(threshold, color='red',
                   linestyle='--',
                   label=f'threshold={threshold}')
        ax.set_title('U-Net output probability distribution')
        ax.set_xlabel('Probability of plug')
        ax.set_ylabel('Pixel count')
        ax.legend(); ax.grid(True)
        _save_fig(fig, sf, 'E_prob_histogram.png')

        # Comparison strip
        strip = _make_strip([
            (gray,      'Original'),
            (prob_vis,  'Probability'),
            (plug_mask, 'Mask'),
            (vis,       'Overlay'),
        ])
        cv2.imwrite(os.path.join(sf,'COMPARISON.png'),strip)

    return plug_mask, prob_full, smooth_top


def enforce_bottom_connected(mask, H):
    """Keep only the region connected to pipe bottom."""
    result = np.zeros_like(mask)
    W = mask.shape[1]
    for x in range(W):
        col = mask[:,x]
        ys  = np.where(col>0)[0]
        if len(ys)==0:
            continue
        # Check if any white pixel is near bottom
        if ys.max() >= H*0.7:
            # Fill from bottom upward to lowest gap
            top = ys.min()
            result[top:H, x] = 255
    return result


def get_smooth_top_boundary(mask, H, W):
    """Extract smooth top boundary from mask."""
    raw = np.full(W, H, dtype=float)
    for x in range(W):
        ys = np.where(mask[:,x]>0)[0]
        if len(ys)>0:
            raw[x] = float(ys.min())

    # Smooth
    win = min(71, W//8)
    if win%2==0: win+=1
    try:
        smooth = savgol_filter(raw, win, 3)
    except Exception:
        smooth = gaussian_filter1d(raw, sigma=10)

    return np.clip(smooth, 0, H-1).astype(int)


# ============================================================
# STEP 3: HEIGHT PROFILE
# ============================================================

def compute_height_profile(smooth_top, H, W, px_per_mm,
                            debug_folder=None,
                            frame_idx=0, save_debug=False):
    heights_mm = np.maximum(0, H-smooth_top) / px_per_mm

    if debug_folder and save_debug:
        sf = os.path.join(debug_folder,
                f'step3_height/frame_{frame_idx:04d}')
        os.makedirs(sf, exist_ok=True)
        fig, ax = plt.subplots(figsize=(13,4))
        xs = np.arange(W)
        ax.fill_between(xs, heights_mm,
                         color='steelblue', alpha=0.4)
        ax.plot(xs, heights_mm, 'b-', lw=1.5)
        ch = _h_center(heights_mm)
        ax.axhline(ch, color='red', linestyle=':',
                    label=f'H_center={ch:.1f}mm')
        ax.axvline(W//2, color='magenta', linestyle='--',
                    label='Center')
        ax.set_xlabel('x (px)'); ax.set_ylabel('Height (mm)')
        ax.set_title(f'Frame {frame_idx}: Height profile')
        ax.legend(); ax.grid(True)
        _save_fig(fig, sf, 'height_profile.png')

    return heights_mm


def _h_center(heights_mm):
    W=len(heights_mm); c=W//2
    w=max(1,W//20)
    return float(heights_mm[c-w:c+w].mean())


# ============================================================
# STEP 4: FRONT ANGLE
# ============================================================

def compute_front_angle(heights_mm, smooth_top, H, W,
                         px_per_mm, pipe_diam_mm,
                         debug_folder=None,
                         frame_idx=0, save_debug=False):
    h_center = _h_center(heights_mm)
    h_ratio  = h_center / pipe_diam_mm

    if h_ratio >= FULL_PLUG_RATIO:
        return None  # full plug, no angle

    # Smooth heights
    h_sm   = gaussian_filter1d(heights_mm, sigma=8)
    grad   = np.gradient(h_sm)

    # Find steepest rising region in left 60% of image
    search_end = int(W * 0.6)
    win_size   = max(15, W//10)

    best_slope  = 0.0
    best_xc     = W//4

    for xs in range(0, search_end-win_size, win_size//3):
        xe = xs + win_size
        if xe > W: break
        coeffs = np.polyfit(np.arange(win_size),
                             h_sm[xs:xe], 1)
        if coeffs[0] > best_slope:
            best_slope = coeffs[0]
            best_xc    = (xs+xe)//2

    if best_slope < 0.02:
        return None

    x0 = max(0, best_xc - win_size//2)
    x1 = min(W, best_xc + win_size//2)
    xs_fit = np.arange(x0, x1, dtype=np.float64)
    ys_fit = h_sm[x0:x1]

    if len(xs_fit) < 5:
        return None

    coeffs    = np.polyfit(xs_fit, ys_fit, 1)
    angle_deg = float(np.degrees(np.arctan(coeffs[0])))

    if debug_folder and save_debug:
        sf = os.path.join(debug_folder,
                f'step4_angle/frame_{frame_idx:04d}')
        os.makedirs(sf, exist_ok=True)
        fig, axes = plt.subplots(2,1, figsize=(13,7))
        xs_arr = np.arange(W)
        ax=axes[0]; ax2=ax.twinx()
        ax.plot(xs_arr, heights_mm,'b-',lw=1.5,
                label='Height')
        ax.plot(xs_arr, h_sm,'c-',lw=1,alpha=0.8,
                label='Smoothed')
        ax2.plot(xs_arr, grad, color='orange',
                 lw=1, alpha=0.6, label='Gradient')
        ax.axvspan(x0,x1,alpha=0.2,color='red',
                   label='Fit region')
        ax.set_ylabel('Height(mm)',color='blue')
        ax2.set_ylabel('Gradient',color='orange')
        ax.set_title(f'Front angle={angle_deg:.1f}° '
                     f'h_ratio={h_ratio:.2f}')
        ax.legend(loc='upper left',fontsize=7)
        ax2.legend(loc='upper right',fontsize=7)
        ax.grid(True)

        y_line = np.polyval(coeffs, xs_fit)
        axes[1].fill_between(xs_arr, heights_mm,
                             color='steelblue',alpha=0.4)
        axes[1].plot(xs_arr,heights_mm,'b-',lw=1)
        axes[1].plot(xs_fit,y_line,'r-',lw=3,
                     label=f'angle={angle_deg:.1f}°')
        axes[1].set_xlabel('x(px)')
        axes[1].set_ylabel('Height(mm)')
        axes[1].legend(); axes[1].grid(True)
        plt.tight_layout()
        _save_fig(fig, sf, 'front_angle.png')

    return angle_deg, best_xc


# ============================================================
# STEP 5: VELOCITY (multi-strip cross-correlation)
# ============================================================

def compute_velocity(frame1, frame2,
                      heights_mm1, heights_mm2,
                      H, W, px_per_mm, effective_fps,
                      n_strips=5,
                      debug_folder=None,
                      frame_idx=0, save_debug=False):
    g1 = _gray(frame1)
    g2 = _gray(frame2)

    h_avg = (_h_center(heights_mm1) +
             _h_center(heights_mm2)) / 2.0
    h_px  = max(10, int(h_avg * px_per_mm))

    strip_h    = max(5, h_px // n_strips)
    velocities = []
    strip_info = []

    for s in range(n_strips):
        y2 = H - s * strip_h
        y1 = max(0, y2 - strip_h)
        if y1 >= y2: continue

        p1 = g1[y1:y2,:].mean(axis=0).astype(float)
        p2 = g2[y1:y2,:].mean(axis=0).astype(float)
        p1 -= p1.mean(); p2 -= p2.mean()

        if p1.std() < 0.5: continue

        corr     = correlate(p2, p1, mode='full')
        shift_px = int(corr.argmax()) - (W-1)
        if abs(shift_px) > W*0.35: continue

        vel = (shift_px / px_per_mm) * effective_fps
        velocities.append(vel)
        strip_info.append({
            'strip': s, 'y1': y1, 'y2': y2,
            'y_mm': (H-(y1+y2)/2)/px_per_mm,
            'vel_mm_s': vel,
        })

    bulk_vel = float(np.median(velocities)) \
               if velocities else 0.0

    if debug_folder and save_debug and strip_info:
        sf = os.path.join(debug_folder,
                f'step5_vel/frame_{frame_idx:04d}')
        os.makedirs(sf, exist_ok=True)

        vis = cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR)
        clrs = [(0,255,0),(0,200,255),(255,128,0),
                (255,0,128),(128,0,255)]
        for si in strip_info:
            c = clrs[si['strip']%len(clrs)]
            cv2.rectangle(vis,(0,si['y1']),(W,si['y2']),c,2)
            cv2.putText(vis,
                        f"{si['vel_mm_s']:.1f}mm/s",
                        (5,(si['y1']+si['y2'])//2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,c,1)
        cv2.putText(vis,f"Bulk={bulk_vel:.1f}mm/s",
                    (10,28),cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(255,255,0),2)
        _save_img(vis,sf,'A_strips.png',
                  'A: Velocity strips')

        y_mms=[s['y_mm']    for s in strip_info]
        vels =[s['vel_mm_s']for s in strip_info]
        fig,ax=plt.subplots(figsize=(6,5))
        ax.plot(vels,y_mms,'bo-',lw=1.5,markersize=8)
        ax.axvline(0,color='k',lw=0.5)
        ax.axvline(bulk_vel,color='blue',linestyle='--',
                   label=f'Bulk={bulk_vel:.1f}')
        ax.set_xlabel('Velocity (mm/s)')
        ax.set_ylabel('Height from bottom (mm)')
        ax.set_title('Layer velocity profile')
        ax.legend(); ax.grid(True)
        _save_fig(fig,sf,'B_profile.png')

    return bulk_vel, strip_info


# ============================================================
# STEP 6: LAYER VELOCITY (optical flow)
# ============================================================

def compute_layer_velocities(frame1, frame2,
                              H, W, px_per_mm,
                              effective_fps,
                              debug_folder=None,
                              frame_idx=0,
                              save_debug=False):
    g1 = _gray(frame1); g2 = _gray(frame2)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        0.5, 3, 21, 3, 5, 1.1, 0
    )
    vx_row = flow[:,:,0].mean(axis=1)
    vx_mms = vx_row / px_per_mm * effective_fps
    y_px   = np.arange(H)
    y_mm   = (H - y_px) / px_per_mm

    if debug_folder and save_debug:
        sf = os.path.join(debug_folder,
                f'step6_layers/frame_{frame_idx:04d}')
        os.makedirs(sf, exist_ok=True)

        mag,ang = cv2.cartToPolar(flow[...,0],flow[...,1])
        hsv=np.zeros((H,W,3),np.uint8)
        hsv[...,0]=ang*180/np.pi/2; hsv[...,1]=255
        hsv[...,2]=cv2.normalize(mag,None,0,255,
                                  cv2.NORM_MINMAX)
        _save_img(cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR),
                  sf,'A_flow_color.png',
                  'Optical flow color')

        fig,ax=plt.subplots(figsize=(5,7))
        ax.plot(vx_mms,y_mm,'b-',lw=1.5)
        ax.fill_betweenx(y_mm,0,vx_mms,
                         alpha=0.3,color='blue')
        ax.axvline(0,color='k',lw=0.5)
        ax.set_xlabel('Velocity (mm/s)')
        ax.set_ylabel('Height from bottom (mm)')
        ax.set_title(f'Frame {frame_idx}: Layer velocity')
        ax.grid(True)
        _save_fig(fig,sf,'B_layer_profile.png')

    return y_mm, vx_mms, flow


# ============================================================
# STEP 7: VOID FRACTION (geometric + Hough)
# ============================================================

def compute_void_fraction(frame, plug_mask,
                           H, W, px_per_mm,
                           bead_diam_mm, pipe_diam_mm,
                           debug_folder=None,
                           frame_idx=0, save_debug=False):
    gray = _gray(frame)
    br   = max(3, int(bead_diam_mm*px_per_mm/2))

    # CLAHE on plug region only
    clahe   = cv2.createCLAHE(clipLimit=3.0,
                                tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    masked   = cv2.bitwise_and(enhanced, enhanced,
                                mask=plug_mask)

    circles = cv2.HoughCircles(
        masked, cv2.HOUGH_GRADIENT, dp=1.0,
        minDist=int(br*1.4),
        param1=40, param2=15,
        minRadius=max(2,br-4), maxRadius=br+5
    )

    circle_list=[]; n_det=0
    if circles is not None:
        for (cx,cy,r) in np.round(circles[0]).astype(int):
            if 0<=cy<H and 0<=cx<W and plug_mask[cy,cx]>0:
                n_det+=1; circle_list.append((cx,cy,r))

    plug_area_mm2 = float((plug_mask>0).sum())/(px_per_mm**2)
    plug_vol_mm3  = plug_area_mm2 * pipe_diam_mm
    bead_vol_mm3  = (4/3)*np.pi*(bead_diam_mm/2)**3
    depth_layers  = pipe_diam_mm/bead_diam_mm
    bead_area_px2 = np.pi*(br**2)
    plug_area_px2 = float((plug_mask>0).sum())

    if n_det>5:
        n_total=int(n_det*depth_layers)
    else:
        n_vis  =max(1,int(plug_area_px2*0.64/bead_area_px2))
        n_total=int(n_vis*depth_layers)

    bead_vol_tot=n_total*bead_vol_mm3
    vf=(float(np.clip(1-bead_vol_tot/plug_vol_mm3,0,1))
        if plug_vol_mm3>0 else 0.36)

    result={
        'void_fraction':    vf,
        'vf_theoretical':   0.36,
        'n_beads_detected': n_det,
        'n_beads_estimated':n_total,
        'plug_area_mm2':    plug_area_mm2,
    }

    if debug_folder and save_debug:
        sf=os.path.join(debug_folder,
                f'step7_void/frame_{frame_idx:04d}')
        os.makedirs(sf,exist_ok=True)
        _save_img(enhanced,sf,'A_CLAHE.png','CLAHE enhanced')
        vis=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        for(cx,cy,r)in circle_list:
            cv2.circle(vis,(cx,cy),r,(0,255,0),1)
            cv2.circle(vis,(cx,cy),2,(0,0,255),-1)
        cv2.putText(vis,
                    f'{n_det} beads  vf={vf:.3f}',
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,255),2)
        _save_img(vis,sf,'B_circles.png',
                  f'B: {n_det} detected beads')

    return result


# ============================================================
# STEP 8: ANNOTATED FRAME
# ============================================================

def visualize_frame(frame, plug_mask, smooth_top,
                     heights_mm, px_per_mm, H, W,
                     frame_idx, void_result,
                     angle=None, velocity=None,
                     strip_info=None,
                     y_mm=None, vx_mms=None,
                     save_path=None,
                     debug_folder=None,
                     save_debug=False):
    gray = _gray(frame)
    vis  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    ov=vis.copy(); ov[plug_mask>0]=(0,160,0)
    vis=cv2.addWeighted(vis,0.5,ov,0.5,0)

    for x in range(W-1):
        y1=int(np.clip(smooth_top[x],  0,H-1))
        y2=int(np.clip(smooth_top[x+1],0,H-1))
        cv2.line(vis,(x,y1),(x+1,y2),(0,0,255),2)

    cx=W//2; ch=_h_center(heights_mm)
    cy_top=int(np.clip(H-ch*px_per_mm,0,H-1))
    cv2.line(vis,(cx,cy_top),(cx,H),(255,0,255),2)
    cv2.putText(vis,f"H={ch:.1f}mm",
                (cx+6,max(18,cy_top+20)),
                cv2.FONT_HERSHEY_SIMPLEX,0.65,
                (255,0,255),2)

    vf=void_result['void_fraction']
    nb=void_result['n_beads_detected']
    lines=[
        f"Frame {frame_idx:04d}",
        f"H_center = {ch:.1f} mm",
        f"Void frac = {vf:.3f}",
        f"Beads det = {nb}",
    ]
    if velocity is not None:
        d='L' if velocity<0 else 'R'
        lines.append(f"Vel = {abs(velocity):.1f} mm/s {d}")
    if angle is not None:
        lines.append(f"Front angle = {angle:.1f} deg")

    for li,txt in enumerate(lines):
        cv2.putText(vis,txt,(10,28+li*28),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,
                    (0,255,255),2,cv2.LINE_AA)

    cv2.arrowedLine(vis,(W-80,H-25),(W-190,H-25),
                    (0,255,255),3,tipLength=0.3)
    cv2.putText(vis,"Flow",(W-185,H-30),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,
                (0,255,255),1)

    if save_path:
        cv2.imwrite(save_path,vis)

    if debug_folder and save_debug:
        sf=os.path.join(debug_folder,
                f'step8_composite/frame_{frame_idx:04d}')
        os.makedirs(sf,exist_ok=True)
        nc=3 if y_mm is not None else 2
        fig,axes=plt.subplots(1,nc,figsize=(6*nc,5))
        axes[0].imshow(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Frame {frame_idx}')
        axes[0].axis('off')
        xs=np.arange(W)
        axes[1].fill_between(xs,heights_mm,
                             color='steelblue',alpha=0.4)
        axes[1].plot(xs,heights_mm,'b-',lw=1.5)
        axes[1].axhline(ch,color='red',linestyle=':',
                        label=f'H={ch:.1f}mm')
        axes[1].set_xlabel('x(px)'); axes[1].set_ylabel('Height(mm)')
        axes[1].legend(); axes[1].grid(True)
        if y_mm is not None:
            axes[2].plot(vx_mms,y_mm,'g-',lw=1.5)
            axes[2].fill_betweenx(y_mm,0,vx_mms,
                                  alpha=0.3,color='green')
            axes[2].axvline(0,color='k',lw=0.5)
            axes[2].set_xlabel('Velocity(mm/s)')
            axes[2].set_ylabel('Height(mm)')
            axes[2].grid(True)
        plt.tight_layout()
        _save_fig(fig,sf,'composite.png')

    return vis


# ============================================================
# MAIN INFERENCE PIPELINE
# ============================================================

def run_inference(frames_folder, output_folder,
                   model_path    = MODEL_PATH,
                   pipe_diam_mm  = PIPE_DIAMETER_MM,
                   bead_diam_mm  = BEAD_DIAMETER_MM,
                   effective_fps = EFFECTIVE_FPS,
                   img_size      = IMG_SIZE,
                   threshold     = 0.5,
                   debug_every_n = 5):

    os.makedirs(output_folder, exist_ok=True)
    ann_dir   = os.path.join(output_folder,'annotated_frames')
    debug_dir = os.path.join(output_folder,'debug_stages')
    os.makedirs(ann_dir,   exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    device = ('cuda' if torch.cuda.is_available()
              else 'cpu')
    print(f"Device: {device}")

    model = load_model(model_path, device)

    paths = sorted(
        glob.glob(os.path.join(frames_folder,'*.png')) +
        glob.glob(os.path.join(frames_folder,'*.jpg'))
    )
    print(f"Found {len(paths)} frames")

    first  = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
    H, W   = first.shape
    px_per_mm = H / pipe_diam_mm
    print(f"Scale: {px_per_mm:.3f} px/mm")

    results=[]; prev_frame=None; prev_heights=None
    y_mm_last=None; vx_last=None

    for i,path in enumerate(paths):
        frame=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        if frame is None: continue
        H,W   = frame.shape
        save_d = debug_every_n>0 and i%debug_every_n==0

        # U-Net segmentation
        plug_mask, prob_map, smooth_top = segment_with_unet(
            frame, model, device, img_size, threshold,
            debug_folder=debug_dir,
            frame_idx=i, save_debug=save_d
        )

        heights_mm = compute_height_profile(
            smooth_top, H, W, px_per_mm,
            debug_folder=debug_dir,
            frame_idx=i, save_debug=save_d
        )
        h_center = _h_center(heights_mm)
        h_ratio  = h_center/pipe_diam_mm

        angle_result = compute_front_angle(
            heights_mm, smooth_top, H, W,
            px_per_mm, pipe_diam_mm,
            debug_folder=debug_dir,
            frame_idx=i, save_debug=save_d
        )
        front_angle = angle_result[0] if angle_result else None

        void_result = compute_void_fraction(
            frame, plug_mask, H, W,
            px_per_mm, bead_diam_mm, pipe_diam_mm,
            debug_folder=debug_dir,
            frame_idx=i, save_debug=save_d
        )

        bulk_vel=None; strip_info=[]
        flow=None; y_mm_now=None; vx_now=None

        if prev_frame is not None:
            bulk_vel, strip_info = compute_velocity(
                prev_frame, frame,
                prev_heights, heights_mm,
                H, W, px_per_mm, effective_fps,
                debug_folder=debug_dir,
                frame_idx=i, save_debug=save_d
            )[:2]
            y_mm_now, vx_now, flow = compute_layer_velocities(
                prev_frame, frame,
                H, W, px_per_mm, effective_fps,
                debug_folder=debug_dir,
                frame_idx=i, save_debug=save_d
            )
            y_mm_last=y_mm_now; vx_last=vx_now

        vis = visualize_frame(
            frame, plug_mask, smooth_top,
            heights_mm, px_per_mm, H, W,
            frame_idx=i,
            void_result=void_result,
            angle=front_angle,
            velocity=bulk_vel,
            strip_info=strip_info,
            y_mm=y_mm_last, vx_mms=vx_last,
            save_path=os.path.join(ann_dir,
                                   f'frame_{i:04d}.png'),
            debug_folder=debug_dir,
            save_debug=save_d
        )

        results.append({
            'frame_index':        i,
            'time_s':             i/effective_fps,
            'height_center_mm':   h_center,
            'height_max_mm':      float(heights_mm.max()),
            'height_ratio':       h_ratio,
            'plug_is_full':       h_ratio>=FULL_PLUG_RATIO,
            'front_angle_deg':    front_angle,
            'bulk_velocity_mm_s': bulk_vel,
            'void_fraction':      void_result['void_fraction'],
            'n_beads_detected':   void_result['n_beads_detected'],
        })

        prev_frame=frame; prev_heights=heights_mm

        v_s=f"{bulk_vel:.1f}" if bulk_vel else "N/A"
        a_s=f"{front_angle:.1f}°" if front_angle else "FULL"
        print(f"Frame {i:04d} | H={h_center:.1f}mm | "
              f"Vel={v_s}mm/s | Angle={a_s} | "
              f"VF={void_result['void_fraction']:.3f}"
              + (" [DBG]" if save_d else ""))

    df=pd.DataFrame(results)
    df.to_csv(os.path.join(output_folder,'results.csv'),
              index=False)
    _plot_summary(df, output_folder, pipe_diam_mm)
    return df


def _plot_summary(df, out, pipe_diam_mm):
    fig,axes=plt.subplots(4,1,figsize=(14,18))
    t=df['time_s']
    axes[0].plot(t,df['height_center_mm'],'b-',lw=1.5,
                 label='Center')
    axes[0].plot(t,df['height_max_mm'],'r--',lw=1,
                 label='Max')
    axes[0].axhline(pipe_diam_mm,color='gray',
                    linestyle=':',label='Pipe diam')
    axes[0].set_ylabel('Height(mm)')
    axes[0].set_title('Plug Height')
    axes[0].legend(); axes[0].grid(True)

    vel=df['bulk_velocity_mm_s'].dropna()
    if len(vel):
        axes[1].plot(df['time_s'][vel.index],vel,'g-',lw=1.5)
        axes[1].axhline(0,color='k',lw=0.5)
    axes[1].set_ylabel('Velocity(mm/s)')
    axes[1].set_title('Bulk Velocity')
    axes[1].grid(True)

    ang=df['front_angle_deg'].dropna()
    if len(ang):
        axes[2].plot(df['time_s'][ang.index],ang,
                     color='orange',lw=1.5)
    axes[2].set_ylabel('Angle(deg)')
    axes[2].set_title('Front Angle')
    axes[2].grid(True)

    axes[3].plot(t,df['void_fraction'],'purple',lw=1.5)
    axes[3].axhline(0.36,color='red',linestyle='--',
                    label='Theory 0.36')
    axes[3].set_ylim(0,1)
    axes[3].set_xlabel('Time(s)')
    axes[3].set_ylabel('Void Fraction')
    axes[3].set_title('Void Fraction')
    axes[3].legend(); axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out,'summary_plot.png'),dpi=150)
    plt.show()


# ── Internal helpers ──────────────────────────────────────────
def _gray(f):
    return (cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
            if len(f.shape)==3 else f.copy())

def _save_img(img,folder,filename,title=None):
    os.makedirs(folder,exist_ok=True)
    d=(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
       if len(img.shape)==2 else img.copy())
    if title:
        cv2.rectangle(d,(0,0),(min(len(title)*11+14,
                                    d.shape[1]),38),(0,0,0),-1)
        cv2.putText(d,title,(6,26),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,
                    (0,255,255),2,cv2.LINE_AA)
    cv2.imwrite(os.path.join(folder,filename),d)

def _save_fig(fig,folder,filename):
    os.makedirs(folder,exist_ok=True)
    fig.savefig(os.path.join(folder,filename),
                dpi=130,bbox_inches='tight')
    plt.close(fig)

def _make_strip(imgs_titles,out_h=250):
    panels=[]
    for img,title in imgs_titles:
        p=(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
           if len(img.shape)==2 else img.copy())
        h,w=p.shape[:2]
        nw=max(1,int(w*out_h/h))
        p=cv2.resize(p,(nw,out_h))
        bar=np.zeros((35,nw,3),np.uint8)
        cv2.putText(bar,title,(4,24),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,
                    (0,255,255),1)
        panels.append(np.vstack([bar,p]))
    mh=max(p.shape[0] for p in panels)
    out=[]
    for p in panels:
        d=mh-p.shape[0]
        if d>0:
            p=np.vstack([p,np.zeros((d,p.shape[1],3),np.uint8)])
        out.append(p)
    return np.hstack(out)


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    run_inference(
        frames_folder = 'frames/',
        output_folder = 'output/',
        model_path    = 'checkpoints/best_model.pth',
        pipe_diam_mm  = 50.0,
        bead_diam_mm  = 3.0,
        effective_fps = 6.0,
        debug_every_n = 5,
    )