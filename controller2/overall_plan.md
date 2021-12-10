# WE DOIN THIS SHIT
## START: at the right-hand corner
```
take off
move to the left a little bit

while has not reached table (aka x-coord not reached the table)
    if no shit in front of us and haven't reached the table
        go forward 
        continue
    if shit in front of us and multiple objects left
        move toward the furthest (region of least optical flow)
        continue
    if shit in front of us and one object left
        if object is too close
            move to the left
            continue
        else
            move forward
            continue

// now we've reached the table

ascend to table height

while haven't landed by the book
    if book is in the middle of the camera
        move forward until it is right in front of us
        land
        DONE
    else if book is at the left of the image
        move to the left
        continue
    else if book is at the right of the image
        move to the right
        continue
    else
        move to the right
        continue
```