arrows = [-0.816300000000000 -0.417100000000000 0.399500000000000;
          -0.754625141663577 -0.470597698280120 0.457161752525532;
          -0.685410341993712 -0.519392708771140 0.510256306786996;
          -0.609347172389336 -0.562997487857721 0.558253159548939;
          -0.527195630897994 -0.600976351001054 0.600672742124214];

figure
scale = 100;
quiver3(r1_origin(1), r1_origin(2), r1_origin(3), scale*r1_axis(1), scale*r1_axis(2), scale*r1_axis(3), 'm', 'lineWidth', 2)
hold on

for i = 1: 5
    scale = 200;
    if i == 1
        quiver3(r1_origin(1), r1_origin(2), r1_origin(3), scale*arrows(i,1), scale*arrows(i,2), scale*arrows(i,3), 'b', 'lineWidth', 2)
        hold on
    else
        quiver3(r1_origin(1), r1_origin(2), r1_origin(3), scale*arrows(i,1), scale*arrows(i,2), scale*arrows(i,3), 'r', 'lineWidth', 2)
        hold on
    end
end
