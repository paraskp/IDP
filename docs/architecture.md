# Dashcam Speed Analyzer — Technical & Algorithmic Architecture

## 1. System Overview & Principle

The Dashcam Speed Analyzer determines the physical velocity of both the ego vehicle (carrying the dashcam) and surrounding traffic from monocular dashcam video without requiring manual user camera calibration.

Monocular computer vision possesses an inherent **metric-scale ambiguity**: a small object close to the camera projects onto the sensor with the exact same pixel footprint as a large object far away. Similarly, camera motion in pixels can represent either high physical speed far from surfaces or low physical speed close to surfaces.

To solve this inverse problem scientifically without fabricating parameters:
1. **Scene Geometry & Ground Plane Constraints**: The camera optical axis intersects the horizontal ground at a pitch angle $\theta$ recoverable from the longitudinal road vanishing point $v_y$.
2. **Ground Contact Point Projection**: The lower edge of a vehicle's bounding box corresponds to the tire contact patch on the road plane $Y = -h_{cam}$.
3. **Multi-Estimator Fusion**: Ground contact depth is fused with physical vehicle class dimensional priors (e.g. standard car width $1.82\text{ m}$).
4. **Honest Confidence Engine**: When geometric constraints are inadequate (e.g. vanishing point not found or uncalibrated camera height), metric scale is marked as unavailable with low confidence rather than fabricating numbers.

---

## 2. Coordinate Frames & Transformations

We define two primary coordinate systems:

### 2.1 Camera Coordinate Frame $\{C\}$
* Origin at the camera optical center.
* $Z_c$: Points forward along the optical axis.
* $X_c$: Points to the right.
* $Y_c$: Points downwards towards the ground.

A 3D point $P_c = [X_c, Y_c, Z_c]^T$ projects to image coordinates $p = [u, v]^T$ via the pinhole camera matrix $K$:
$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim K \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$

### 2.2 Road Ground Frame $\{W\}$
* Plane $Y_w = 0$ corresponds to the flat road surface.
* The camera is mounted at height $h_{cam}$ above the road surface ($Y_c = -h_{cam}$).
* Camera pitch angle $\theta$ is the angle between the optical axis and the road surface:
  $$\theta = \arctan\left(\frac{v_y - c_y}{f_y}\right)$$
  where $v_y$ is the vertical coordinate of the longitudinal vanishing point.

---

## 3. Detailed Algorithmic Stages

### Stage 1: Video Ingestion & Validation
* Probes video container with `OpenCV` to verify file integrity and extract stream properties ($W, H, FPS, N_{frames}, \text{FOURCC}$).
* Calculates exact timestamps $t_i = i / FPS$ and human-readable timecodes.

### Stage 2: Camera & Scene Calibration
* Extracts linear segments with Canny edge detection and Probabilistic Hough Transform (`cv2.HoughLinesP`).
* Discards near-horizontal noise lines.
* Employs RANSAC consensus over line pair intersections to estimate the principal vanishing point $VP = (v_x, v_y)$.
* Computes road horizon line and ground normal vector $n_{ground}$.
* If two orthogonal vanishing points exist, calculates exact focal length $f = \sqrt{-(v_{1x}-c_x)(v_{2x}-c_x) - (v_{1y}-c_y)(v_{2y}-c_y)}$. Otherwise, flags FOV as uncalibrated prior with honest low confidence.

### Stage 3: Stationary Feature Detection
* Shi-Tomasi corner detection (`cv2.goodFeaturesToTrack`) within the drivable road corridor and environmental margins.
* Dynamically masks out bounding boxes of detected moving vehicles to prevent tracking other cars as environmental landmarks.

### Stage 4: Visual Odometry & Optical Flow
* Pyramidal Lucas-Kanade optical flow with bidirectional forward-backward validation:
  $$||p_{prev} - p_{prev\_backward}|| \le \epsilon_{fb} \quad (\epsilon_{fb} = 1.5\text{ px})$$
* RANSAC Essential Matrix estimation ($E = K^T F K$) and pose recovery via `cv2.recoverPose` to isolate ego-rotation $R$ and translation direction $t$.

### Stage 5: Ego Speed Estimation
* Road plane depth for stationary road texture points:
  $$Z_i = \frac{f_y \cdot h_{cam}}{y_i - v_y}$$
* Forward displacement $\Delta Z = Z_{t} - Z_{t+\Delta t}$ over time step $\Delta t$ yields instantaneous speed $v = \Delta Z / \Delta t$.
* 1D kinematic Kalman Filter ($[x, v, a]^T$) smooths road texture noise and produces continuous velocity and acceleration.

### Stage 6 & 7: Vehicle Detection & ByteTrack
* YOLOv8 neural detector detects vehicles filtered to COCO classes: car, truck, bus, motorcycle.
* ByteTrack two-stage association matches high-confidence and low-confidence detections using IoU cost matrix, maintaining persistent track IDs through temporary occlusions and motion blur.

### Stage 8: 3D Depth Estimation
* Vehicle contact patch $(u_{bottom}, v_{bottom})$ projected to ground plane:
  $$Z_{ground} = \frac{f_y \cdot h_{cam}}{v_{bottom} - v_y}, \quad X_{ground} = \frac{(u_{bottom} - c_x) \cdot Z_{ground}}{f_x}$$
* Fused with dimensional depth prior ($Z_{dim} = \frac{f_x \cdot W_{prior}}{w_{bbox}}$) via adaptive weighting.

### Stage 9: Other Vehicle Speed Estimation
* Vehicle trajectory history $(X_t, Z_t)$ differentiated over a sliding temporal window ($\Delta t \approx 0.5\text{s}$) via linear regression:
  $$\vec{v}_{rel} = \left(\frac{dX}{dt}, \frac{dZ}{dt}\right)$$
* Absolute ground speed:
  $$\vec{v}_{abs} = \vec{v}_{rel} + \begin{bmatrix} 0 \\ v_{ego} \end{bmatrix}, \quad \text{Speed} = ||\vec{v}_{abs}||$$
* Classifies motion direction into: `same_direction`, `opposite_direction`, `crossing`, or `stationary`.

### Stage 10: Annotation & Results Generation
* Overlays telemetry HUD, vehicle labels, speed tags, confidence ratings, and trajectory trails onto processed video.
* Exports detailed per-frame CSV logs, comprehensive JSON session summaries, and Matplotlib speed-vs-time trajectory plots.
