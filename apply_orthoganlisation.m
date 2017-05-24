function data_org = apply_orthoganlisation(data_input)
addpath('/Users/au194693/projects/tutorials/osl_workshop/osl-core');
osl_startup;

data_input
data = load(data_input);
data = data.data;
data_org = zeros(size(data));

for j=1:size(data, 1)
    data_org(j, :, :) = ROInets.remove_source_leakage(squeeze(data(j, :, :)), 'closest');
end

save('data_org', 'data_org')
exit
x