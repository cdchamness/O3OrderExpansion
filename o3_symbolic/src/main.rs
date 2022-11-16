mod bra_ket;
mod inner_product;
mod kdelta;
mod term;

use crate::bra_ket::*;
use crate::inner_product::*;
use crate::term::*;

fn main() {
    let basic_bra = BraKet::new('l', 'y', vec![0]);
    let basic_ket = BraKet::new('l', 'y', vec![1]);

    let basic_ip = InnerProduct::new(1.0, basic_bra, Vec::new(), basic_ket, None);

    let mut new_term = Term::new(vec![basic_ip]);
    let lapp = new_term.lappalacian('x', 'a');
    for mut t in lapp {
        t.sort_ips();
        println!("{}", t);
    }

    println!("\nDoing Gradiant Product Test\n");

    let bk1 = BraKet::new('l', 'y', vec![0, 0]);
    let bk2 = BraKet::new('l', 'y', vec![1, 0]);
    let bk3 = BraKet::new('l', 'y', vec![0, 1]);

    let ip1 = InnerProduct::new(1.0, bk1.clone(), vec![], bk2.clone(), None);
    let ip2 = InnerProduct::new(1.0, bk1.clone(), vec![], bk3.clone(), None);

    let t1 = Term::new(vec![ip1.clone()]);
    let t2 = Term::new(vec![ip2.clone()]);

    for t in t1.gradiant_product(t2, 'x', 'a') {
        println!("{}", t);
    }

    println!("\n\nDoing parity_transform Test:\n");
    let mut t3 = Term::new(vec![ip1, ip2]);
    t3.reduce();
    println!("{}", t3);
    t3.parity_transform(0);
    t3.shift_down();
    t3.sort_ips();
    println!("{}", t3);
    t3.parity_transform(1);
    t3.shift_down();
    t3.sort_ips();
    println!("{}", t3);
    t3.parity_transform(0);
    t3.shift_down();
    t3.sort_ips();
    println!("{}", t3);
    t3.parity_transform(1);
    t3.shift_down();
    t3.sort_ips();
    println!("{}", t3);
    t3.parity_reduce(vec![true,true]);
    println!("{}", t3);


    println!("\nDoing shift_down test: \n");

    let bk4 = BraKet::new('l', 'y', vec![0, -1]);
    let bk5 = BraKet::new('l', 'y', vec![-1, -2]);
    let ip3 = InnerProduct::new(1.0, bk4, vec![], bk5, None);
    let mut t3 = Term::new(vec![ip3]);
    println!("{}", t3);
    t3.shift_down();
    println!("{}", t3);
    if let Some(mut new_ip) = t3.get_ips().pop() {
        new_ip.order_bra_kets();
        println!("{}", new_ip);
    }
}
