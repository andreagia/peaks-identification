
fetch 5EH8
hide everything
show cartoon
color grey40
bg_color white

select hetatm and not solvent
show stick, sele
color green, sele
copy pred, 5EH8

select highauto, (resi 29+125) and 5EH8
show spheres, highauto
set sphere_scale, 1.0, highauto
color red, highauto

select lowauto, (resi 63+94+127+130+143+199+245) and 5EH8
show spheres, lowauto
set sphere_scale, 1.0, lowauto
color pink, lowauto

select highexp, (resi 63+66+68+88+94+96+117+127+128+130+134+143+199+245) and 5EH8
show spheres, highexp
set sphere_scale, 1.0, highexp
color red, highexp

select lowexp, (resi 62+90+93+97+98+123+129+140+142+145+193+195+203+204+208+209+211+244) and 5EH8
show spheres, lowexp
set sphere_scale, 1.0, lowexp
color pink, lowexp

deselect