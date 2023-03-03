use crate::kdelta::*;
use std::cmp::{Ord, Ordering, PartialOrd};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BraKet {
    lattice_type: char,
    index_type: char,
    shift: Vec<i8>,
}

impl BraKet {
    pub fn new(lattice_type: char, index_type: char, shift: Vec<i8>) -> BraKet {
        BraKet {
            lattice_type,
            index_type,
            shift,
        }
    }

    pub fn get_string(&self) -> String {
        let mut my_str = self.index_type.to_string();
        my_str += "_";
        for (i, step) in self.shift.iter().enumerate() {
            if i != 0 {
                my_str += ",";
            }
            my_str += &step.to_string();
        }
        my_str.to_string()
    }

    pub fn get_shift(&self) -> Vec<i8> {
        self.shift.clone()
    }

    pub fn extend_shift_len(&mut self) {
        self.shift.push(0);
    }

    pub fn partial(&self, partial_index_type: char) -> Option<KDelta> {
        if partial_index_type == self.index_type {
            let sum = self.shift.iter().fold(0, |acc, x| acc + x.abs());
            if sum % 2 != 0 {
                return None;
            }
        }
        Some(KDelta::new(
            self.index_type,
            partial_index_type,
            self.shift.clone(),
        ))
    }

    pub fn collapse_delta(&mut self, delta: &KDelta) {
        if delta.index_type_1 != delta.index_type_2 {
            if self.index_type == delta.index_type_1
                || self.index_type == delta.index_type_2 && self.shift.len() == delta.shift.len()
            {
                self.index_type = if self.index_type == delta.index_type_1 {
                    delta.index_type_2
                } else {
                    delta.index_type_1
                };
                self.shift = (0..self.shift.len())
                    .map(|i| self.shift[i] - delta.shift[i])
                    .collect();
            }
        } else if self.index_type == delta.index_type_1 && self.shift.len() == delta.shift.len() {
            if let Some(last) = delta.get_last_non_zero_index() {
                while self.shift[last] != 0 {
                    for i in 0..self.shift.len() {
                        self.shift[i] -= delta.shift[i]
                    }
                }
                self.shift.remove(last);
            }
        }
    }

    pub fn get_neg_shifts(&self) -> Vec<(i8, usize)> {
        let mut out = Vec::new();
        for (index, shift_count) in self.shift.iter().enumerate() {
            if shift_count < &0 {
                out.push((-shift_count, index));
            }
        }
        out
    }

    pub fn do_shift(&mut self, shift_amount: i8, index: usize) {
        self.shift[index] += shift_amount;
    }

    pub fn parity_transform(&mut self, index: usize) {
        self.shift[index] = -self.shift[index];
    }
}

impl PartialOrd for BraKet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let size = self.shift.len();
        match size.cmp(&other.shift.len()) {
            Ordering::Equal => {
                for i in 0..size {
                    match self.shift[i].cmp(&other.shift[i]) {
                        Ordering::Less => {
                            return Some(Ordering::Less);
                        }
                        Ordering::Greater => {
                            return Some(Ordering::Greater);
                        }
                        Ordering::Equal => {}
                    }
                }
                Some(Ordering::Equal)
            }
            Ordering::Greater => Some(Ordering::Greater),
            Ordering::Less => Some(Ordering::Less),
        }
    }
}

impl Ord for BraKet {
    fn cmp(&self, other: &Self) -> Ordering {
        let size = self.shift.len();
        match size.cmp(&other.shift.len()) {
            Ordering::Equal => {
                for i in 0..size {
                    match self.shift[i].cmp(&other.shift[i]) {
                        Ordering::Less => {
                            return Ordering::Less;
                        }
                        Ordering::Greater => {
                            return Ordering::Greater;
                        }
                        Ordering::Equal => {}
                    }
                }
                Ordering::Equal
            }
            Ordering::Greater => Ordering::Greater,
            Ordering::Less => Ordering::Less,
        }
    }
}

impl fmt::Display for BraKet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut disp_string = "|".to_string();
        disp_string += &self.get_string();
        disp_string += ">";

        write!(f, "{}", disp_string)
    }
}

#[cfg(test)]
mod tests {
    use crate::bra_ket::*;

    #[test]
    fn collapse_x_delta_1() {
        let mut bra = BraKet::new('l', 'x', vec![0, 1, 1]);
        let delta = KDelta::new('x', 'y', vec![0, 1, 0]);
        bra.collapse_delta(&delta);
        assert_eq!(bra, BraKet::new('l', 'y', vec![0, 0, 1]))
    }

    #[test]
    fn collapse_x_delta_2() {
        let mut bra = BraKet::new('l', 'x', vec![0, 1]);
        let delta = KDelta::new('x', 'y', vec![1, 1]);
        bra.collapse_delta(&delta);
        assert_eq!(bra, BraKet::new('l', 'y', vec![-1, 0]))
    }

    #[test]
    fn collapse_x_delta_3() {
        let mut bra = BraKet::new('l', 'x', vec![0, 1]);
        let delta = KDelta::new('y', 'z', vec![1, 1]);
        bra.collapse_delta(&delta);
        assert_eq!(bra, BraKet::new('l', 'x', vec![0, 1]))
    }

    #[test]
    fn collapse_mu_delta_1() {
        let mut bra = BraKet::new('l', 'x', vec![0, 1]);
        let delta = KDelta::new('x', 'x', vec![1, 1]);
        bra.collapse_delta(&delta);
        assert_eq!(bra, BraKet::new('l', 'x', vec![-1]))
    }

    #[test]
    fn collapse_mu_delta_2() {
        let mut bra = BraKet::new('l', 'x', vec![0, 1, 0, 1, 0]);
        let delta = KDelta::new('x', 'x', vec![0, 0, 1, 1, 0]);
        bra.collapse_delta(&delta);
        assert_eq!(bra, BraKet::new('l', 'x', vec![0, 1, -1, 0]))
    }

    #[test]
    fn collapse_mu_delta_3() {
        let mut bra = BraKet::new('l', 'x', vec![0, 1, 0, 1, 0]);
        let delta = KDelta::new('y', 'y', vec![0, 0, 1, 1, 0]);
        bra.collapse_delta(&delta);
        assert_eq!(bra, BraKet::new('l', 'x', vec![0, 1, 0, 1, 0]))
    }

    #[test]
    fn collapse_mu_delta_4() {
        let mut bra = BraKet::new('l', 'x', vec![0, 1, 1, 3, 0]);
        let delta = KDelta::new('x', 'x', vec![0, 0, 1, 1, 0]);
        bra.collapse_delta(&delta);
        assert_eq!(bra, BraKet::new('l', 'x', vec![0, 1, -2, 0]))
    }

    #[test]
    fn collapse_mu_delta_5() {
        let mut bra = BraKet::new('l', 'x', vec![0, 1, 1, 0, 0]);
        let delta = KDelta::new('x', 'x', vec![0, 0, 1, 1, 0]);
        bra.collapse_delta(&delta);
        assert_eq!(bra, BraKet::new('l', 'x', vec![0, 1, 1, 0]))
    }
}
