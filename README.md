# Quadrotor Attitude Control — MPC and Nonlinear MPC

Attitude control of a quadrotor about hover, from an undergraduate nonlinear control
course. Hover is an unstable equilibrium and the plant is nonlinear, underactuated and
fast — the rotational inertias are on the order of $10^{-4}$ kg·m², so the interesting
dynamics happen in milliseconds.

## The plant

Rotational dynamics under body torques $\tau_{x,y,z}$, combining Euler's equations with
Euler-angle kinematics:

$$
\begin{aligned}
I_x\dot\omega_x &= \tau_x + (I_y - I_z)\,\omega_y\omega_z - d_x\omega_x \\
I_y\dot\omega_y &= \tau_y + (I_z - I_x)\,\omega_z\omega_x - d_y\omega_y \\
I_z\dot\omega_z &= \tau_z + (I_x - I_y)\,\omega_x\omega_y - d_z\omega_z \\
\dot\phi &= \omega_x + \sin\phi\tan\theta\;\omega_y + \cos\phi\tan\theta\;\omega_z \\
\dot\theta &= \cos\phi\;\omega_y - \sin\phi\;\omega_z \\
\dot\psi &= \tfrac{\sin\phi}{\cos\theta}\,\omega_y + \tfrac{\cos\phi}{\cos\theta}\,\omega_z
\end{aligned}
$$

where $\phi,\theta,\psi$ are roll, pitch and yaw, and $\omega_{x,y,z}$ are body-frame
rates.

| Parameter | Value | Units | Meaning |
|---|---|---|---|
| $I_x, I_y$ | $6\times10^{-5}$ | kg·m² | Roll, pitch inertia |
| $I_z$ | $1.2\times10^{-4}$ | kg·m² | Yaw inertia |
| $d_x, d_y$ | $1\times10^{-5}$ | N·m·s | Roll, pitch aerodynamic damping |
| $d_z$ | $2\times10^{-5}$ | N·m·s | Yaw aerodynamic damping |

## Requirements

**Constraint** — rotor torque is limited to $|\tau| \le 0.25$ N·m.

**Objectives**

- Track step references in roll and pitch up to $\pm10°$, with rise time under 1 s and
  overshoot under 5%.
- Reject a torque impulse of up to 0.1 N·m with less than 1° of deviation from setpoint
  and recovery within 1 s.
- Remain robust to parametric uncertainty in the inertias and damping coefficients.

## Tasks

### Part A — modelling and linearisation

1. Choose a state and input vector and write the nonlinear state-space equations
   $\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u})$ as a minimal realisation.
2. Linearise about the hover equilibrium $(\mathbf{x}^\ast, \mathbf{u}^\ast) = (0, 0)$,
   give $\mathbf{A}$ and $\mathbf{B}$ both parametrically and numerically, and determine
   whether the linearised plant is stable.

### Part B — linear MPC

1. Design an MPC for the linearised plant with full state feedback ($\mathbf{C} = \mathbf{I}_6$)
   that meets the objectives and respects the torque constraint. Justify the prediction
   and control horizons, sample time, weights, and the signal-type choices.
2. Simulate 15 s of closed loop: initial condition $\phi = 5°$, $\theta = -5°$, with a
   step reference to $\phi = 10°$, $\theta = -10°$ applied at $t = 10$ s. Plot the control
   actions and all three angles.
3. Simulate 4 s of closed loop with an impulse disturbance
   $[\tau_x, \tau_y, \tau_z] = [0.1, -0.1, 0.05]$ N·m applied at $t = 1$ s. Plot the
   control actions and angles.
4. Build uncertain plants by perturbing $I_{x,y,z}$ by 55% and $d_{x,y,z}$ by 75% about
   nominal. Using the *same* controller, repeat the Part B-2 scenario for 15 random
   samples and discuss the resulting performance.

### Part C — the linear controller on the nonlinear plant

Write a function returning the nonlinear state derivatives, and a second function that
integrates them with the forward Euler method. Apply the Part B controller to this
nonlinear plant, simulate tracking of $\phi = 10°$, $\theta = -10°$, and compare against
the linear results from B-2.

### Part D — nonlinear MPC

Design an NMPC for the nonlinear plant. Simulate both the step-tracking and the impulse
disturbance scenarios, plot control actions and angles, justify the design parameters, and
assess whether performance, robustness and stability improve on the linear controller
of Part C.
