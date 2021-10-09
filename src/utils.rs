use num_traits::{Num, Float};

pub fn get_distance<P: Float + Num + PartialOrd + Clone>(p1: &[P; 2], p2: &[P; 2]) -> P{
    let dx = p2[0]-p1[0];
    let dy = p2[1]-p1[1];
    let distance = dx*dx+dy*dy;
    distance.sqrt()
}