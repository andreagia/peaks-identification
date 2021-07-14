
fetch 5T8R
hide everything
show cartoon
color grey40
bg_color white

select hetatm and not solvent
show stick, sele
color green, sele
copy pred, 5T8R

select highauto, (resi 1675+1676+1689+1692) and pred
show spheres, highauto
set sphere_scale, 1.0, highauto
color red, highauto

select lowauto, (resi 1674+1696+1703) and pred
show spheres, lowauto
set sphere_scale, 1.0, lowauto
color pink, lowauto

select highexp, (resi 1675+1677+1693) and 5T8R
show spheres, highexp
set sphere_scale, 1.0, highexp
color red, highexp

select lowexp, (resi 1674+1688+1689+1691+1692+1696+1703) and 5T8R
show spheres, lowexp
set sphere_scale, 1.0, lowexp
color pink, lowexp

deselect