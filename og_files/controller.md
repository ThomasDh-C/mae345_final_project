1. RED filter - works

current:
- take off and yeet to target
- PROBLEM: duuuh obstacles

fixes for obstacles:
- Slowly move forwards and see LHS obstacle's optical flow vs RHS obstacle's optical flow
- move at 45 in direction of least resistance (given bounds of course)
    - How to do bounds of course? Perhaps filter lower half of image for white, then do some quick mafs to figure out how far from the center we start 
    - Other option: Use same starting point each time and keep
    track of movement in x direction

fixes for random yeeting:
- function for nice maneuvers that are slow
- no matter direction of movement, steady velocity we can set + pauses at start and end

fixes for ending (book detection):
- book detection and landing
    - test books, find one that works
    - distance to book


TDC's wish:
the white lines


KT - book detection. testing bayybbbeeeee
BELLA - random yeet function
TDC - program structure + optical flow on obstacles
JACOB - white lines (don't take that out of context)