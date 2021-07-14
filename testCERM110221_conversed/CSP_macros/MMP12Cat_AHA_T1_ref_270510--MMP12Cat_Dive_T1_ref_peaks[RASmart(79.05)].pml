
fetch 4GQL
hide everything
show cartoon
color grey40
bg_color white

select hetatm and not solvent
show stick, sele
color green, sele
copy pred, 4GQL

select highauto, (resi 179+233+235+241) and 4GQL
show spheres, highauto
set sphere_scale, 1.0, highauto
color red, highauto

select lowauto, (resi 176+178+180+211+218+223+230+234+236+240+242) and 4GQL
show spheres, lowauto
set sphere_scale, 1.0, lowauto
color pink, lowauto

select highexp, (resi 183+220+223+235+236+241+242) and 4GQL
show spheres, highexp
set sphere_scale, 1.0, highexp
color red, highexp

select lowexp, (resi 178+179+180+213+240) and 4GQL
show spheres, lowexp
set sphere_scale, 1.0, lowexp
color pink, lowexp

deselect