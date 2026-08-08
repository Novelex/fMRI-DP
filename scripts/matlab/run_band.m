function run_band(work_dir, funimg_source_dir, subject_ids_csv, low_cutoff, high_cutoff, config_file, parallel_workers)
% Run DPARSFA detrend+ALFF for one frequency band across a list of subjects.
%
% work_dir           - DPARSFA working directory for this band
% funimg_source_dir  - directory containing <FILE_ID>_func_preproc.nii.gz files to symlink in
% subject_ids_csv    - comma-separated list of subject FILE_IDs
% low_cutoff         - CalALFF.AHighPass_LowCutoff (Hz)
% high_cutoff        - CalALFF.ALowPass_HighCutoff (Hz)
% config_file        - path to save the Cfg .mat file
% parallel_workers   - Cfg.ParallelWorkersNumber (default 1 if omitted)

if nargin < 7
    parallel_workers = 1;
end

project_root = '/users/3171356m/muhammad/GraSTIACL';
spm12_dir = fullfile(project_root, 'data/software/SPM12');
dpabi_dir = fullfile(project_root, 'data/software/DPABI');
mask_file = fullfile(dpabi_dir, 'Templates', 'BrainMask_05_61x73x61.img');

subject_ids = strsplit(subject_ids_csv, ',');

funimg_dir = fullfile(work_dir, 'FunImg');
if ~isfolder(funimg_dir)
    mkdir(funimg_dir);
end

for i = 1:numel(subject_ids)
    sid = subject_ids{i};
    subj_dir = fullfile(funimg_dir, sid);
    if ~isfolder(subj_dir)
        mkdir(subj_dir);
    end
    source_file = fullfile(funimg_source_dir, [sid, '_func_preproc.nii.gz']);
    link_file = fullfile(subj_dir, [sid, '_func_preproc.nii.gz']);
    if ~isfile(link_file)
        system(['ln -sf ', source_file, ' ', link_file]);
    end
end

subject_list_file = fullfile(work_dir, 'SubjectList.txt');
fid = fopen(subject_list_file, 'w');
for i = 1:numel(subject_ids)
    fprintf(fid, '%s\n', subject_ids{i});
end
fclose(fid);

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
Cfg.ParallelWorkersNumber = parallel_workers;
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
Cfg.CalALFF.AHighPass_LowCutoff = low_cutoff;
Cfg.CalALFF.ALowPass_HighCutoff = high_cutoff;

save(config_file, 'Cfg');
disp('Config saved:');
disp(config_file);

[Error, AutoDataProcessParameter] = DPARSFA_run(Cfg, work_dir, subject_list_file, 0);

result_file = strrep(config_file, '.mat', '_result.mat');
save(result_file, 'Error', 'AutoDataProcessParameter');
disp('DPARSFA error output:');
disp(Error);

end
