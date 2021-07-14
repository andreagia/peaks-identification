
fetch 1Z3J
hide everything
show cartoon
color grey40
bg_color white

select hetatm and not solvent
show stick, sele
color green, sele
copy pred, 1Z3J

select highauto, (resi 147+180+183+203+206+207+210+215+220+239+241) and pred
show spheres, highauto
set sphere_scale, 1.0, highauto
color red, highauto

select lowauto, (resi 178+184+213+224+236) and pred
show spheres, lowauto
set sphere_scale, 1.0, lowauto
color pink, lowauto

select highexp, (resi 110+111+112+185+199+201+203+204+205) and 1Z3J
show spheres, highexp
set sphere_scale, 1.0, highexp
color red, highexp

select lowexp, (resi 121+147+202+207+239+263) and 1Z3J
show spheres, lowexp
set sphere_scale, 1.0, lowexp
color pink, lowexp

deselect