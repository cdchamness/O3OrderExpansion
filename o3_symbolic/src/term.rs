use crate::inner_product::*;
use std::ops::Mul;

pub struct Term {
    ips: Vec<InnerProduct>
}

impl Term {

    pub fn new(ips: Vec<InnerProduct>) -> Term {
        Term { ips }
    }
}


impl Mul<InnerProduct> for Term {
    type Output = Term;

    fn mul(self, rhs: InnerProduct) -> Term {
        let mut ips = self.ips.clone();
        ips.push(rhs);
        Term::new(ips)
    }
}