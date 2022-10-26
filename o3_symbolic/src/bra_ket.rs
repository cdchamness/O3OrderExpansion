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
    	if self.index_type == delta.index_type_1 {
    		if self.shift.len() == delta.shift.len() {
    			let last = delta.get_last_non_zero_index();
    			if self.shift[last] != 0 {
    				let mut new_shift = Vec::new();
    				for (i, step) in self.shift.iter().enumerate() {
    					new_shift.push(delta.shift[i] - self.shift[i])
    				}
    				return BraKet::new(self.lattice_type, delta.index_type_2, new_shift);
    			}
    		}
    	}
    	if self.index_type == delta.index_type_2 {
    		if self.shift.len() == delta.shift.len() {
    			let last = delta.get_last_non_zero_index();
    			if self.shift[last] != 0 {
    				let mut new_shift = Vec::new();
    				for (i, step) in self.shift.iter().enumerate() {
    					new_shift.push(delta.shift[i] - self.shift[i])
    				}
    				return BraKet::new(self.lattice_type, delta.index_type_1, new_shift);
    			}
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