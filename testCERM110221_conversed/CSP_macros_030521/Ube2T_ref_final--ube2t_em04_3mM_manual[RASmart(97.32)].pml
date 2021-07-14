
fetch 5NGZ
hide everything
show cartoon
color grey40
bg_color white

select hetatm and not solvent
show stick, sele
color green, sele
copy pred, 5NGZ

select highauto, (resi 74+76+85+86+119+124) and pred
show spheres, highauto
set sphere_scale, 1.0, highauto
color red, highauto

select lowauto, (resi 8+24+25+37+82+83+84+87+104+154) and pred
show spheres, lowauto
set sphere_scale, 1.0, lowauto
color pink, lowauto

select highexp, (resi 16+74+85+86) and 5NGZ
show spheres, highexp
set sphere_scale, 1.0, highexp
color red, highexp

select lowexp, (resi 8+76+82+83+104+119+124) and 5NGZ
show spheres, lowexp
set sphere_scale, 1.0, lowexp
color pink, lowexp

deselect