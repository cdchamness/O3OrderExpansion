use crate::tools;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KDelta {
    pub index_type_1: char,
    pub index_type_2: char,
    pub shift: Vec<i8>,
}

impl KDelta {
    pub fn new(index_type_1: char, index_type_2: char, shift: Vec<i8>) -> KDelta {
        KDelta {
            index_type_1,
            index_type_2,
            shift,
        }
    }

    pub fn get_last_non_zero_index(&self) -> Option<usize> {
        Some(self.shift.len() - self.shift.iter().rev().position(|&r| r != 0)? - 1)
    }

    pub fn get_all_non_zero_indicies(&self) -> Vec<usize> {
        self.shift
            .iter()
            .enumerate()
            .flat_map(|(index, &value)| {
                if value != 0 {
                    vec![index; value.abs() as usize]
                } else {
                    vec![]
                }
            })
            .collect()
    }

    pub fn collapse_delta(&mut self, delta: &KDelta) {
        if self.index_type_1 == self.index_type_2
            && delta.index_type_1 == delta.index_type_2
            && self.index_type_1 == delta.index_type_1
        {
            // All index_types are the same
            if self.shift.len() == delta.shift.len() {
                // KDeltas are over the same number of summations
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
    }

    pub fn expand_delta(&self) -> Vec<Vec<KDelta>> {
        if self.index_type_1 == self.index_type_2 {
            let sum = self.shift.iter().fold(0, |acc, x| acc + x.abs());
            if sum % 2 == 0 && sum >= 4 {
                // These are the conditions that must occur for this step to be needed
                let mut non_zero_indicies = self.get_all_non_zero_indicies();
                let pairings = tools::gen_pairs(&mut non_zero_indicies);
                let mut out: Vec<Vec<KDelta>> = Vec::new();
                for pairing in pairings {
                    let mut kdeltas: Vec<KDelta> = Vec::new();
                    for pair in pairing {
                        let (a, b) = pair;
                        let mut new_shift = vec![0; self.shift.len()];
                        new_shift[a] = self.shift[a].signum();
                        new_shift[b] = self.shift[b].signum();
                        let kd = KDelta {
                            index_type_1: self.index_type_1,
                            index_type_2: self.index_type_2,
                            shift: new_shift,
                        };
                        kdeltas.push(kd);
                    }
                    out.push(kdeltas);
                }

                return out;
            }
        }
        // if the conditions to expand are not met, just wrap self in necessary vecs
        vec![vec![self.clone()]]
    }
}

impl fmt::Display for KDelta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut my_str = "δ(".to_string();
        my_str += &self.index_type_1.to_string();
        my_str += "|";
        my_str += &self.index_type_2.to_string();
        my_str += "_";
        for (i, shift) in self.shift.iter().enumerate() {
            if i != 0 {
                my_str += ",";
            }
            my_str += &shift.to_string();
        }
        my_str += ")";
        write!(f, "{}", my_str)
    }
}

#[cfg(test)]
mod tests {
    use crate::kdelta::*;

    #[test]
    fn expand_test1() {
        let index_type_1 = 'x';
        let index_type_2 = 'x';
        let kd1 = KDelta {
            index_type_1,
            index_type_2,
            shift: vec![1, 1, 1, 1],
        };

        let mut expanded: Vec<Vec<KDelta>> = Vec::new();

        let expanded_kd1 = KDelta {
            index_type_1,
            index_type_2,
            shift: vec![1, 1, 0, 0],
        };
        let expanded_kd2 = KDelta {
            index_type_1,
            index_type_2,
            shift: vec![0, 0, 1, 1],
        };
        expanded.push(vec![expanded_kd1, expanded_kd2]);

        let expanded_kd3 = KDelta {
            index_type_1,
            index_type_2,
            shift: vec![1, 0, 1, 0],
        };
        let expanded_kd4 = KDelta {
            index_type_1,
            index_type_2,
            shift: vec![0, 1, 0, 1],
        };
        expanded.push(vec![expanded_kd3, expanded_kd4]);

        let expanded_kd5 = KDelta {
            index_type_1,
            index_type_2,
            shift: vec![1, 0, 0, 1],
        };
        let expanded_kd6 = KDelta {
            index_type_1,
            index_type_2,
            shift: vec![0, 1, 1, 0],
        };
        expanded.push(vec![expanded_kd5, expanded_kd6]);

        assert_eq!(kd1.expand_delta(), expanded);
    }
}
