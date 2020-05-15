IN_FILE=~/database/M/linsi/M1_M2_all_trim.fasta
OUT_FILE=~/database/M/linsi/M1_M2_filtered_trim.fasta
CSV_FILE=~/database/M/linsi/M1_M2_all_trim_combination.csv
OUT_dir=2005061715/


#python filter_identical_sequence.py $IN_FILE $OUT_FILE



mkdir $OUT_dir
python training.py --fasta_path $OUT_FILE --output_dir $OUT_dir --times_of_training 3 --eps 0.02  --combine_info $CSV_FILE

