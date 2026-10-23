# Simulation Tool: MuJoCo

We will use **MuJoCo** to simulate this task.

## Why MuJoCo

- **Many small balls.** The liquid is made of many small balls touching each other and the container walls. MuJoCo handles this kind of contact fast and stays stable.
- **Easy to measure.** With the `mujoco` Python package we can read every ball's position, so it is easy to count balls in the tube, in the source, or spilled.
- **Fast.** It runs faster than real time, which helps when we test and tune the pouring again and again.
- **Ready robot arms.** MuJoCo Menagerie has ready arm models (for example Franka Panda and UR5e).
- **Free** and well maintained.

## Why Not Gazebo

Gazebo is good when a project depends on ROS 2. But its physics gets slow and shaky with hundreds of small balls touching each other, and we do not need ROS for this version.

## Things to Watch Out For

1. **Hollow containers.** MuJoCo treats a mesh as a solid block for collisions. So we build tubes, beakers and flasks from simple parts (a base plus thin wall pieces), or split the mesh into smaller convex pieces.
2. **Small time step.** Fast small balls can pass through thin walls. Use a small time step (about 1–2 ms) and walls that are not too thin.
3. **Ball size.** Smaller balls look more like liquid but make the simulation slower. We tried 5 mm balls (about 2,000 of them) and it ran 25 times slower than real time. We use 7 mm balls (about 400). See [simulation-environment.md](../../simulation-environment/docs/simulation-environment.md) for this and the other settings we had to find.
