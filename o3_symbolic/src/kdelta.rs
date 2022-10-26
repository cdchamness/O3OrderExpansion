use std::fmt;

#[derive(Clone, Debug, PartialEq)]
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

	pub fn to_string(&self) -> String {
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
        my_str
	}

	pub fn get_last_non_zero_index(&self) -> usize {
		self.shift.len() - self.shift.iter().rev().position(|&r| r != 0).unwrap() - 1
	}
}


impl fmt::Display for KDelta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let disp_string = self.to_string();

        write!(f, "{}", disp_string)
    }
}