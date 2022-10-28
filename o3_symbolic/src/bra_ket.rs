use std::fmt;
use crate::kdelta::*;


#[derive(Clone, Debug, PartialEq)]
pub struct BraKet {
    lattice_type: char,
    index_type: char,
    shift: Vec<i8>,
}

impl BraKet {
    pub fn new(lattice_type: char, index_type: char, shift: Vec<i8>) -> BraKet {
        BraKet { lattice_type, index_type, shift }
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

    pub fn partial(&self, partial_index_type: char) -> Option<KDelta> {

    	if partial_index_type == self.index_type {
    		let sum = self.shift.iter().fold(0, |acc, x| acc + x.abs());
    		if sum % 2 != 0 {
    			return None;
    		}
    	}
    	Some(KDelta::new(self.index_type, partial_index_type, self.shift.clone()))
    }
    
    pub fn collapse_delta(&self, delta: &KDelta) -> BraKet {
		if delta.index_type_1 != delta.index_type_2 {
			if self.index_type == delta.index_type_1 || self.index_type == delta.index_type_2 && self.shift.len() == delta.shift.len() {
				let new_index = if self.index_type == delta.index_type_1 {
					delta.index_type_2
				} else {
					delta.index_type_1
				};
				let new_shift: Vec<i8> = (0..self.shift.len()).map(|i| self.shift[i] - delta.shift[i]).collect();
				return BraKet::new(self.lattice_type, new_index, new_shift);
			}
		}
		else if self.index_type == delta.index_type_1 && self.shift.len() == delta.shift.len() {
			if let Some(last) = delta.get_last_non_zero_index() {	
				let mut new_shift = self.shift.clone();
				for _j in 0..self.shift[last]{
					for i in 0..self.shift.len() {
						new_shift[i] -= delta.shift[i]
					}
				};
				new_shift.remove(last);
				return BraKet::new(self.lattice_type, self.index_type, new_shift);
			}
		}
    	self.clone()
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
	use crate::kdelta::*;
	use crate::bra_ket::*;

	#[test]
	fn collapse_x_delta_1() {
		let bra = BraKet::new('l', 'x', vec![0,1,1]);
		let delta = KDelta::new('x', 'y', vec![0,1,0]);
		assert_eq!(bra.collapse_delta(&delta), BraKet::new('l', 'y', vec![0,0,1]))
	}

	#[test]
	fn collapse_x_delta_2() {
		let bra = BraKet::new('l', 'x', vec![0,1]);
		let delta = KDelta::new('x', 'y', vec![1,1]);
		assert_eq!(bra.collapse_delta(&delta), BraKet::new('l', 'y', vec![-1,0]))
	}

	#[test]
	fn collapse_x_delta_3() {
		let bra = BraKet::new('l', 'x', vec![0,1]);
		let delta = KDelta::new('y', 'z', vec![1,1]);
		assert_eq!(bra.collapse_delta(&delta), BraKet::new('l', 'x', vec![0,1]))
	}

	#[test]
	fn collapse_mu_delta_1() {
		let bra = BraKet::new('l', 'x', vec![0,1]);
		let delta = KDelta::new('x', 'x', vec![1,1]);
		assert_eq!(bra.collapse_delta(&delta), BraKet::new('l', 'x', vec![-1]))
	}

	#[test]
	fn collapse_mu_delta_2() {
		let bra = BraKet::new('l', 'x', vec![0,1,0,1,0]);
		let delta = KDelta::new('x', 'x', vec![0,0,1,1,0]);
		assert_eq!(bra.collapse_delta(&delta), BraKet::new('l', 'x', vec![0,1,-1,0]))
	}

	#[test]
	fn collapse_mu_delta_3() {
		let bra = BraKet::new('l', 'x', vec![0,1,0,1,0]);
		let delta = KDelta::new('y', 'y', vec![0,0,1,1,0]);
		assert_eq!(bra.collapse_delta(&delta), BraKet::new('l', 'x', vec![0,1,0,1,0]))
	}

	#[test]
	fn collapse_mu_delta_4() {
		let bra = BraKet::new('l', 'x', vec![0,1,1,3,0]);
		let delta = KDelta::new('x', 'x', vec![0,0,1,1,0]);
		assert_eq!(bra.collapse_delta(&delta), BraKet::new('l', 'x', vec![0,1,-2,0]))
	}

	#[test]
	fn collapse_mu_delta_5() {
		let bra = BraKet::new('l', 'x', vec![0,1,1,0,0]);
		let delta = KDelta::new('x', 'x', vec![0,0,1,1,0]);
		assert_eq!(bra.collapse_delta(&delta), BraKet::new('l', 'x', vec![0,1,1,0]))
	}
}
