#[derive(Clone)]
pub struct BraKet {
    lattice_type: char,
    index_type: char,
    shifts: Vec<i8>,
}

impl BraKet {
    pub fn new(lattice_type: char, index_type: char, shifts: Vec<i8>) -> BraKet {
        BraKet { lattice_type, index_type, shifts }
    }
}