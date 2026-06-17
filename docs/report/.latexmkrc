use Cwd;
# Build artifacts go in build/; the shared bibliography lives in ../assets.
$out_dir = 'build';
# Absolute BIBINPUTS so bibtex finds references.bib regardless of its CWD
# (bibtex runs inside $out_dir when -outdir/$out_dir is set).
$ENV{'BIBINPUTS'} = getcwd() . '/../assets' . ($ENV{'BIBINPUTS'} ? ':' . $ENV{'BIBINPUTS'} : '') . ':';
