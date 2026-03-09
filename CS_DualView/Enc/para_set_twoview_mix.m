function [measure_1, measure_2] = para_set_twoview_mix(measure, measure_1, measure_2)


measure_1.rate_allocation = ceil((measure_1.image_width*measure_1.image_height+measure_2.image_width*measure_2.image_height)*measure.rate);
measure_2.rate_allocation = measure_1.rate_allocation;
measure_1.OMEGA = 1:measure_1.rate_allocation;
measure_2.OMEGA = 1:measure_2.rate_allocation;
[A_1, At_1, measure_1] = Measure_matrix_create(measure_1, measure_1.seed);
measure_1.A = A_1;
measure_1.At = At_1;


[A_2, At_2, measure_2] = Measure_matrix_create(measure_2, measure_2.seed);
measure_2.A = A_2;
measure_2.At = At_2;

end
