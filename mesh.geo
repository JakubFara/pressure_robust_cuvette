h = 0.02;
Point(1) = { -1, -1, 0, h};
Point(2) = { 1,  -1, 0, h};
Point(3) = { 1,  1, 0, h};
Point(4) = { -1, 1, 0, h};
Point(5) = { -0.5, -0.5, 0, h};
Point(6) = { 0.5,  -0.5, 0, h};
Point(7) = { 0.5,  0.5, 0, h};
Point(8) = { -0.5, 0.5, 0, h};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Curve Loop( 9) = {1, 2, 3, 4};
Curve Loop(10) = {5, 6, 7, 8};
Plane Surface(1) = {9, 10};

Physical Curve("outer_bottom", 11) = {1};
Physical Curve("outer_left", 12) = {2};
Physical Curve("outer_top", 13) = {3};
Physical Curve("outer_bottom", 14) = {4};

Physical Curve("outer_bottom", 21) = {5};
Physical Curve("outer_left", 22) = {6};
Physical Curve("outer_top", 23) = {7};
Physical Curve("outer_bottom", 24) = {8};

Physical Surface("surface", 1) = {1};
