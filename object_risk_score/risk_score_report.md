# Object Risk Score Algorithm Report

This report outlines the exact inputs, mathematical formulas, and algorithmic logic used by the `risk_score_calculator.py` engine to determine the threat level of surrounding objects.

---

## 1. The Inputs
The prediction engine takes raw kinematic data for each tracked object. For every single frame, it ingests the following parameters:
*   **$X$**: Longitudinal distance from the ego vehicle (meters).
*   **$Y$**: Lateral distance from the ego vehicle (meters).
*   **$V_x$**: Relative longitudinal velocity (meters/second).
*   **$V_y$**: Relative lateral velocity (meters/second).

---

## 2. The Core Calculations (Algorithm)

Before calculating the final score, the engine derives three critical physical metrics to understand exactly how the object is behaving relative to the ego vehicle (which is always sitting at the origin `(0,0)`).

### A. Absolute Distance ($D$)
The total direct-line distance between the object and the ego vehicle.
> **Formula:** $D = \sqrt{X^2 + Y^2}$

### B. Radial Velocity ($V_{rad}$)
Instead of just looking at forward speed, the engine calculates the **Radial Velocity**—which is the exact velocity component pointing *directly at* the ego vehicle.
> **Formula:** $V_{rad} = \frac{(X \cdot V_x) + (Y \cdot V_y)}{D}$
*   **Relevance:** If a car is moving fast but driving *parallel* to you, it's not a threat. Radial velocity isolates only the movement vector that could cause a collision. A negative $V_{rad}$ means the object is actively approaching.

### C. Time-to-Collision ($TTC$)
If the object is approaching ($V_{rad} < -0.1$ m/s), the engine predicts exactly how many seconds remain until impact.
> **Formula:** $TTC = \frac{D}{|V_{rad}|}$
*   **Relevance:** This is the gold standard metric in ADAS (Advanced Driver Assistance Systems). A car 100 meters away going 50 m/s is vastly more dangerous than a car 5 meters away going 0 m/s. 

---

## 3. The Final Risk Score Formula

The final risk score is a blended percentage (capped at 100%) that weights both spatial proximity and temporal urgency.

> **Formula:** 
> $Risk = \left( \frac{100}{D + 1.0} \right) + \left( \frac{50}{TTC + 0.1} \right)$

### Why this specific formula?
1. **The Proximity Penalty:** $\left( \frac{100}{D + 1.0} \right)$
   * Even if a vehicle is completely stationary ($TTC = \infty$), if it is extremely close to your car, it remains an inherent risk. The `+1.0` is a mathematical safeguard to prevent "division by zero" errors if an object perfectly overlaps the ego vehicle.
2. **The Urgency Penalty:** $\left( \frac{50}{TTC + 0.1} \right)$
   * This is the heavy hitter. If an object has a TTC of 5 seconds, it adds a mild 10% risk. But if the TTC drops to 0.5 seconds, it instantly adds 83% risk, sending the total score skyrocketing to 100%. The `+0.1` ensures that sub-second impact times don't break the math engine.
3. **The 100% Cap:**
   * Any combined score over 100 is simply capped at `100.0%`, indicating an imminent, unavoidable critical threat.

---

> [!TIP]
> **Tuning the Engine:**
> If you find the ego vehicle is too "nervous" (reacting to things too early), you can easily lower the `50` numerator in the TTC penalty. If it reacts too late, you can increase it to `75` or `100`.
