union() {
    for(i = [-1, 0, 1]) {
        for (j = [-1, 0, 1]) {
            for (k = [-1, 0, 1]) {
                translate([i, j, k]) {
                    rotate([90, 0, 0]) cylinder(h=1, r=0.1, center=true, $fn=100);
                    rotate([0, 90, 0]) cylinder(h=1, r=0.1, center=true, $fn=100);
                    cylinder(h=1, r=0.1, center=true, $fn=100);
                }
            }
        }
    }
}