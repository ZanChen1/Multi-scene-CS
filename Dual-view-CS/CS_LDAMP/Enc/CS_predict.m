function [par]=CS_predict(y, OMEGA, par, measure, distr, bit)

if measure.lowrank_preprocess==1
    predict_image_dir = ['../predict_image/',distr,'/preprocess_1_',num2str(measure.c1),'/',measure.Image_name(1:end-4),'/',strrep(num2str(bit),' ',''),'/'];
else
    predict_image_dir = ['../predict_image/',distr,'/nonpreprocess_1/',measure.Image_name(1:end-4),'/',strrep(num2str(bit),' ',''),'/'];
end
if ~exist(predict_image_dir,'dir')
    mkdir(predict_image_dir);
end

if iscell(OMEGA)
    OMEGA=cell2mat(OMEGA);
end
rate = length(OMEGA);
filename = [predict_image_dir,strrep(num2str(roundn(rate,-4)),'  ','_'),'.mat'];
rate = rate/(measure.image_width*measure.image_height)/2;
if exist(filename,'file')
    load(filename);
else
    par.model=measure.model;
    par=Set_parameters_dec(par,rate,par.model);
    [rec_im]   =  NLR_CS_Reconstruction_predict(y, OMEGA, par, measure);
    save(filename, 'rec_im');
end
par.predict_im=rec_im;





end