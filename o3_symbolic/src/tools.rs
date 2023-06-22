pub fn gen_all_permuations(n: usize) -> Vec<Vec<usize>> {
    let mut base: Vec<usize> = (0..n).collect();
    let mut perms = Vec::new();
    gen_perms(n, &mut base, &mut perms);
    perms
}

fn gen_perms(n: usize, base: &mut Vec<usize>, perms: &mut Vec<Vec<usize>>) {
    if n == 1 {
        perms.push(base.clone());
        return;
    }

    for i in 0..n {
        base.swap(i, n - 1);
        gen_perms(n - 1, base, perms);
        base.swap(i, n - 1);
    }
}
