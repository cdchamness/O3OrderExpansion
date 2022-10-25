mod bra_ket;
mod inner_product;
mod term;

use crate::bra_ket::*;
use crate::inner_product::*;
use crate::term::*;


fn main() {
    let basic_bra = BraKet::new('l', 'x', vec![0]);
    let basic_ket = BraKet::new('l', 'x', vec![1]);
    
    let basic_ip = InnerProduct::new(1.0, basic_bra, "".to_string(), basic_ket, Vec::new(), Vec::new());
}