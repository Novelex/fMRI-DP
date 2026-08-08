load('/users/3171356m/muhammad/GraSTIACL/data/dparsf_work/pilot_configs/Cfg_multitr_classical_result.mat');
disp('Top-level fields:');
disp(fieldnames(AutoDataProcessParameter));
if isfield(AutoDataProcessParameter, 'TR')
    disp('Cfg.TR after run:');
    disp(AutoDataProcessParameter.TR);
end
if isfield(AutoDataProcessParameter, 'SliceTiming')
    disp('SliceTiming field:');
    disp(AutoDataProcessParameter.SliceTiming);
end
if isfield(AutoDataProcessParameter, 'TRSet')
    disp('TRSet field:');
    disp(AutoDataProcessParameter.TRSet);
end
