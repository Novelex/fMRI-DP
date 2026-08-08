% Pilot ALFF run: 1 subject (SDSU_0050182, TR=2.0s), classical band (0.010-0.080 Hz).
% Data is already realigned/normalized by C-PAC, so only detrend + ALFF are enabled.
% Cfg.TR = 0 lets DPARSFA auto-detect each subject's TR from its own NIfTI header.

project_root = '/users/3171356m/muhammad/GraSTIACL';
spm12_dir = fullfile(project_root, 'data/software/SPM12');
dpabi_dir = fullfile(project_root, 'data/software/DPABI');
work_dir = fullfile(project_root, 'data/dparsf_work/pilot_classical');
subject_list_file = fullfile(work_dir, 'SubjectList.txt');
mask_file = fullfile(dpabi_dir, 'Templates', 'BrainMask_05_61x73x61.img');
config_file = fullfile(project_root, 'data/dparsf_work/pilot_configs/Cfg_classical_pilot.mat');

addpath(spm12_dir);
addpath(genpath(dpabi_dir));
spm('defaults', 'fmri');
spm_jobman('initcfg');

Cfg = struct();
Cfg.WorkingDir = work_dir;
Cfg.DataProcessDir = work_dir;
Cfg.SubjectID = {};
Cfg.FunctionalSessionNumber = 1;
Cfg.StartingDirName = 'FunImg';
Cfg.TR = 0;
Cfg.ParallelWorkersNumber = 1;
Cfg.IsAllowGUI = 0;

Cfg.IsNeedConvertFunDCM2IMG = 0;
Cfg.IsNeedConvertT1DCM2IMG = 0;
Cfg.IsBIDStoDPARSF = 0;
Cfg.IsApplyDownloadedReorientMats = 0;
Cfg.RemoveFirstTimePoints = 0;
Cfg.IsSliceTiming = 0;
Cfg.IsRealign = 0;
Cfg.IsCalVoxelSpecificHeadMotion = 0;
Cfg.IsNeedReorientFunImgInteractively = 0;
Cfg.IsNeedReorientCropT1Img = 0;
Cfg.IsNeedReorientT1ImgInteractively = 0;
Cfg.IsBet = 0;
Cfg.IsAutoMask = 0;
Cfg.IsNeedT1CoregisterToFun = 0;
Cfg.IsNeedReorientInteractivelyAfterCoreg = 0;
Cfg.IsSegment = 0;
Cfg.IsDARTEL = 0;
Cfg.IsCovremove = 0;
Cfg.IsFilter = 0;
Cfg.IsNormalize = 0;
Cfg.IsSmooth = 0;
Cfg.IsScrubbing = 0;

Cfg.IsCalReHo = 0;
Cfg.IsCalDegreeCentrality = 0;
Cfg.IsCalFC = 0;
Cfg.IsExtractROISignals = 0;
Cfg.IsExtractAALTC = 0;
Cfg.IsCalVMHC = 0;
Cfg.IsCWAS = 0;

Cfg.MaskFile = mask_file;
Cfg.IsWarpMasksIntoIndividualSpace = 0;

Cfg.IsDetrend = 1;
Cfg.IsCalALFF = 1;
Cfg.CalALFF.AHighPass_LowCutoff = 0.010;
Cfg.CalALFF.ALowPass_HighCutoff = 0.080;

save(config_file, 'Cfg');
disp('Pilot classical config saved:');
disp(config_file);
disp('Mask file:');
disp(mask_file);
disp(['Mask exists: ', num2str(exist(mask_file, 'file') == 2)]);

[Error, AutoDataProcessParameter] = DPARSFA_run(Cfg, work_dir, subject_list_file, 0);

result_file = fullfile(project_root, 'data/dparsf_work/pilot_configs/Classical_pilot_result.mat');
save(result_file, 'Error', 'AutoDataProcessParameter');

disp('DPARSFA error output:');
disp(Error);
