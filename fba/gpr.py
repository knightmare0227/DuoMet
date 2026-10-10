import pandas as pd
from Bio import SeqIO
import os
import fnmatch
import subprocess


def process_reaction_and_gene_data(rxn_file, reaction_file, fasta_file, folder_path, diamond_path):
    rxn_df = pd.read_csv(rxn_file)
    reaction_df = pd.read_csv(reaction_file).rename(columns={'reaction': 'rxn_id'})
    mapped_df = rxn_df.merge(reaction_df, on='rxn_id', how='left').fillna('None')

    if 'model' in mapped_df.columns and 'gene' in mapped_df.columns:
        mapped_df['merged'] = mapped_df['model'] + '.' + mapped_df['gene']
    else:
        raise ValueError("Input file must contain 'model' and 'gene' columns.")

    gene_ids = set(mapped_df['merged'].drop_duplicates())
    extracted_sequences = []
    sequence_dict = {}

    for record in SeqIO.parse(fasta_file, 'fasta'):
        if record.id in gene_ids:
            extracted_sequences.append(record)
            sequence_dict[record.id] = str(record.seq)

    mapped_df['sequence'] = mapped_df['merged'].map(sequence_dict)

    for root, dirs, files in os.walk(folder_path):
        for filename in fnmatch.filter(files, '*.fasta'):
            fasta_file_path = os.path.join(root, filename)
            reference_db = f"{os.path.splitext(filename)[0]}"
            output_db_name = f"{diamond_path}_{reference_db}"

            db_command = f"{diamond_path} makedb --in {fasta_file_path} -d {output_db_name}"
            subprocess.run(db_command, shell=True, check=True)

            output_file = os.path.join(folder_path, 'score.tsv')
            blast_command = f"{diamond_path} blastp -d {output_db_name} -q {fasta_file} -o {output_file}"
            os.system(blast_command)

            tsv_df = pd.read_csv(output_file, sep='\t', header=None)
            mapped_df['gene_id'] = None

            for index, row in mapped_df.iterrows():
                merged_value = row['merged']
                matched_rows = tsv_df[tsv_df[0] == merged_value]

                if not matched_rows.empty:
                    highest_row = matched_rows.loc[matched_rows[2].idxmax()]
                    mapped_df.at[index, 'gene_id'] = highest_row[1]

    mapped_df.fillna('None', inplace=True)

    mapped_df = mapped_df[mapped_df['gene_id'].astype(bool)]
    final_df = mapped_df.drop_duplicates(subset='rxn_id', keep='first')

    final_df = final_df[['rxn_id', 'gene_id', 'gene', 'sequence']]

    final_df.fillna('none', inplace=True)

    output_csv_file = f'/hpcfs/fhome/lian/123456y7uu8/test/{os.path.splitext(filename)[0]}_最终结果.csv'  #### 结果输出路径，需要与下面输出路径一起修改
    final_df.to_csv(output_csv_file, index=False)

    print(f"Output saved to {output_csv_file}")


def process_csv(input_file, output_file):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 对 'rxn_id' 列去重并合并 'gene_id' 列
    df = df.groupby('rxn_id').agg({
        'gene_id': lambda x: ','.join(x.unique()),  # 合并 'gene_id'，用逗号分隔
        **{col: 'first' for col in df.columns if col not in ['rxn_id', 'gene_id']}  # 保留其他列的第一行内容
    }).reset_index()

    # 保存处理后的数据到新文件
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    # 路径

    rxn_file = "./GCF_000005845.2_ASM584v2.csv"  ####上传总反应的csv，反应列列名需修改为rxn_id
    reaction_file = "./gpr_files/bigg_MetaCYC_gprs.csv"  # gpr数据库
    fasta_file = "./gpr_files/DB_CARVEME.fasta"  # 序列数据库
    folder_path = "./gpr_files/fasta_files/"  ####上传你的fasta文件

    for root, dirs, files in os.walk(folder_path):
        for filename in fnmatch.filter(files, '*.fasta'):
            fasta_file_path = os.path.join(root, filename)
            reference_db = f"{os.path.splitext(filename)[0]}"
    output_csv_file = f'./{os.path.splitext(filename)[0]}_final_results.csv'  ####结果输出路径
    diamond_path = "/hpcfs/fhome/lian/123456y7uu8/diamond"  # DIAMOND 安装路径

    # 调用
    process_reaction_and_gene_data(rxn_file, reaction_file, fasta_file, folder_path, diamond_path)
    # 调用函数
    process_csv(output_csv_file, output_csv_file)