# v2-capping

v2 of the lab task: **screw caps on the tubes**, which come off only by turning. For now each of the
three receivers (the empty tubes) has a 24-410 screw cap; the sources do not yet.

It builds on v1 (`../v1-liquid-pouring/`), using v1's lab scene, arms and tubes unchanged (v1's files are
never edited), and adds the caps to the scene before it is built.

**Read in this order:**

1. [docs/capping.md](docs/capping.md): the cap and the physics of how it comes off (diagrams).
2. [docs/uncapping-programmed.md](docs/uncapping-programmed.md): two arms take the cap off, every move
   hand-written (pictures, a GIF).
3. [docs/uncapping-learning.md](docs/uncapping-learning.md): ACT learns the unscrewing from what the arm
   feels; why ACT, what data it gets, the results.

Each has a `-details.md` page beside it with the exact settings and numbers, for when you need them.
[docs/autobio-uncapping.md](docs/autobio-uncapping.md) is a side study of another project's uncapping.

```
make setup in ../v1-liquid-pouring/simulation-environment first (v1's models), then here:

make view        the capped scene in MuJoCo's window
make render      pictures of it from every camera
make cap-trial   the cap opened by the real arm and gripper, in quarter turns (about 30 s)
make uncap       one uncapping run with both arms (about 50 s); SEED=150 SIZE=0.2
make uncap-eval  uncapping on scenes 150-159, 200 mm (10 runs, about 3 min); SIZES="0.15 0.2" for both
make uncap-stress  200 mm tube, scenes 200-229, random placement and cap tightness (about 7 min)
make grip-test   how much twist Arm A's grip on the tube holds, for each grip setting (about 1 min)
make test        the tests
make assets      download AutoBio's screw-cap tubes for later (docs/capping-details.md, section 7)
make autobio-tubes   those tubes on our thread, caps on and off (pictures)
```

Two rules for any code using a capped scene: step it with `caps.step(model, data)`, never
`mujoco.mj_step`, and reset it with `capping.scene.reset(model, data)`.
