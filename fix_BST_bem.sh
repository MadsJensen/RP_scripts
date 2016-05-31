

source ~/.bashrc

cd /scratch2/MINDLAB2011_24-MEG-readiness/fs_subjects_dir/$1/bem

mris_convert head_bem_12962V.fs head_bem_12962V.stl
mris_convert innerskull_bem_12962V.fs innerskull_bem_12962V.stl
mris_convert outerskull_bem_12962V.fs outerskull_bem_12962V.stl

meshfix rh.head_bem_12962V.stl -u 10 --vertices 4098 --fsmesh
meshfix rh.innerskull_bem_12962V.stl -u 10 --vertices 4098 --fsmesh
meshfix rh.outerskull_bem_12962V.stl -u 10 --vertices 4098 --fsmesh

mv rh.innerskull_bem_12962V_fixed.fsmesh inner_skull.surf
mv rh.outerskull_bem_12962V_fixed.fsmesh outer_skull.surf
mv rh.head_bem_12962V_fixed.fsmesh outer_skin.surf

rm rh*.stl
