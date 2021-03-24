function distance=inner_product(px, py, pz, v)

pt = [px, py, pz];
distance = sum(abs(pt * v') / (norm(pt) / norm(v, 'fro')));
end