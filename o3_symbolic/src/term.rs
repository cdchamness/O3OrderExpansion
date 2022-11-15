use std::fmt;
use std::ops::{Add, Mul};

use crate::inner_product::*;
use crate::kdelta::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Term {
    ips: Vec<InnerProduct>,
}

impl Term {
    pub fn new(ips: Vec<InnerProduct>) -> Term {
        Term { ips }
    }

    pub fn scalar_reduce(&mut self) {
        let mut accum = 1.0;
        for ip in &mut self.ips {
            accum *= ip.extract_scalar();
        }
        self.ips[0] *= accum
    }

    pub fn get_ips(&self) -> Vec<InnerProduct> {
        self.ips.clone()
    }

    pub fn duplicate(&self) -> Term {
        let mut ips = Vec::new();
        for ip in &self.ips {
            ips.push(ip.clone())
        }
        Term::new(ips)
    }

    pub fn partial(&mut self, partial_index_type: char, alpha_type: char) -> Vec<Term> {
        let mut out = Vec::new();
        let ip_count = self.ips.len();
        for _ in 0..ip_count {
            let ip = self.ips.pop().unwrap();
            let d_ips = ip.partial(partial_index_type, alpha_type);
            for d_ip in d_ips {
                out.push(d_ip * self.duplicate());
            }
            self.ips.insert(0, ip);
        }

        out
    }

    pub fn lapalacian(&mut self, partial_index_type: char, alpha_type: char) -> Vec<Term> {
        let mut lapp = Vec::new();
        let d_terms = self.partial(partial_index_type, alpha_type);
        for mut dt in d_terms {
            dt.collapse_all_deltas();
            let dd_terms = dt.partial(partial_index_type, alpha_type);
            for mut ddt in dd_terms {
                ddt.scalar_reduce();
                ddt.collapse_all_deltas();
                let out = ddt.alpha_reduce(alpha_type);
                for mut t in out {
                    t.scalar_reduce();
                    lapp.push(t);
                }
            }
        }
        lapp
    }

    pub fn collapse_delta(&mut self, delta: &KDelta) {
        for ip in &mut self.ips {
            ip.collapse_delta(delta);
        }
    }

    pub fn collapse_all_deltas(&mut self) {
        while let Some(next_delta) = self.get_next_delta() {
            self.collapse_delta(&next_delta);
        }
    }

    fn get_next_delta(&self) -> Option<KDelta> {
        for ip in &self.ips {
            match ip.get_delta() {
                Some(delta) => return Some(delta),
                None => {}
            }
        }
        None
    }

    fn get_alpha_count(&self, alpha: char) -> Vec<usize> {
        let mut count: Vec<usize> = Vec::new();
        for ip in &self.ips {
            let mut counts = 0;
            for gen in ip.get_inner() {
                if gen.get_type() == alpha {
                    counts += 1
                }
            }
            count.push(counts);
        }
        count
    }

    pub fn alpha_reduce(&mut self, alpha: char) -> Vec<Term> {
        let inner_count = self.get_alpha_count(alpha);
        if inner_count.iter().sum::<usize>() == 2 {
            // if there are 2 inners, either they are on the same InnerProduct or they are split
            if let Some(max_value) = inner_count.iter().max() {
                if *max_value == 2 {
                    // They are both on the same InnerProduct
                    let index = inner_count.iter().position(|&x| x == 2).unwrap();
                    self.ips[index].clear_inner_alpha(alpha);
                    self.ips[index] *= -2.0;

                    // duplicate self, change ips at index to new_ip
                    vec![self.duplicate()]
                } else {
                    // They are split across multiple InnerProducts
                    let mut alpha_indices = Vec::new();
                    let mut other_indices = Vec::new();
                    for (i, x) in inner_count.iter().enumerate() {
                        if *x == 1 {
                            alpha_indices.push(i);
                        } else {
                            other_indices.push(i);
                        }
                    }

                    let mut other_ips = Vec::new();
                    for index in other_indices {
                        other_ips.push(self.ips[index].clone())
                    }
                    let other_term = Term { ips: other_ips };

                    // indices should have 2
                    let ip1 = &self.ips[alpha_indices[0]];
                    let ip2 = &self.ips[alpha_indices[1]];

                    let (term1, term2) = ip1.collapse_alpha(ip2, alpha);

                    let mut out = Vec::new();
                    out.push(other_term.duplicate() * term1);
                    out.push(other_term * term2);
                    out
                }
            } else {
                // Should not be possible to ever reach
                unreachable!("max is only called if the sum == 2 => there must be a max");
            }
        } else {
            // if there are not 2 'inners' of the same type this does nothing.
            vec![self.duplicate()]
        }
    }
}

impl Mul<InnerProduct> for Term {
    type Output = Term;

    fn mul(self, rhs: InnerProduct) -> Term {
        let mut ips = self.ips;
        ips.push(rhs);
        Term::new(ips)
    }
}

impl Mul<f64> for Term {
    type Output = Term;

    fn mul(self, rhs: f64) -> Term {
        let mut ips = self.ips;
        let mut front = ips.remove(0);
        front *= rhs;
        ips.insert(0, front);
        Term::new(ips)
    }
}

impl Mul<Term> for Term {
    type Output = Term;

    fn mul(self, rhs: Term) -> Term {
        let mut ips = self.ips;
        for new_ip in rhs.get_ips() {
            ips.push(new_ip);
        }
        Term::new(ips)
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut disp_string = "".to_string();
        for ip in &self.ips {
            disp_string += &ip.to_string();
        }

        write!(f, "{}", disp_string)
    }
}

impl Add<Term> for Term {
    type Output = Vec<Term>;

    fn add(self, rhs: Term) -> Vec<Term> {
        if self == rhs {
            let mut cl = rhs.clone();
            cl.scalar_reduce();
            let scalar = cl.get_ips()[0].extract_scalar();
            vec![self * scalar]
        } else {
            vec![self, rhs]
        }
    }
}
