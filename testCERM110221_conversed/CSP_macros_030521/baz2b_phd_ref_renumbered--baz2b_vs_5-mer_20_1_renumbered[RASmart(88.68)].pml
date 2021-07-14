
fetch 6FHQ
hide everything
show cartoon
color grey40
bg_color white

select hetatm and not solvent
show stick, sele
color green, sele
copy pred, 6FHQ

select highauto, (resi 1947+1951+1955) and pred
show spheres, highauto
set sphere_scale, 1.0, highauto
color red, highauto

select lowauto, (resi 1943+1944+1946+1949+1954+1971+1972) and pred
show spheres, lowauto
set sphere_scale, 1.0, lowauto
color pink, lowauto

select highexp, (resi 1947+1948) and 6FHQ
show spheres, highexp
set sphere_scale, 1.0, highexp
color red, highexp

select lowexp, (resi 1944+1946+1949+1951+1955+1971+1972) and 6FHQ
show spheres, lowexp
set sphere_scale, 1.0, lowexp
color pink, lowexp

deselect