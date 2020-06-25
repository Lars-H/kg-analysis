import pandas as pd
from tqdm import tqdm


def read_edglist(classname, outdir):
    df = pd.read_csv(f"{outdir}/{classname}/{classname}.g.csv")  # , names=['t', 'b'])
    return df


def p_t(df):
    return len(df['b'].unique())


def i_t_d(df):
    return len(df['t'].unique())


def oc_p_i_t_d(p, df):
    return len(df[df['b'] == p])


def compute_coverage_slow(df):
    predicates = df['b'].unique()
    oc_sum = 0
    for predicate in tqdm(predicates):
        oc_sum += oc_p_i_t_d(predicate, df)
    return oc_sum / (p_t(df) * i_t_d(df))


def compute_coverage(df):
    g_df = df.groupby(['b']).count()
    oc_sum = sum(g_df.values)
    denom = p_t(df) * i_t_d(df)
    return oc_sum / denom


def compute_coherence(classes, outdir):
    coverage_dct = {}
    p_t_dct = {}
    i_t_d_dct = {}
    for clss in tqdm(classes):
        df = read_edglist(clss, outdir)
        coverage_dct[clss] = compute_coverage(df)
        p_t_dct[clss] = p_t(df)
        i_t_d_dct[clss] = i_t_d(df)

    denominator = sum(p_t_dct.values()) + sum(i_t_d_dct.values())

    coherence = 0
    for clss, coverage in coverage_dct.items():
        wt = (p_t_dct[clss] + i_t_d_dct[clss]) / denominator
        coherence += wt * coverage
        print(f"{clss}:  {coverage}, {wt}")
    return coherence


if __name__ == '__main__':
    outdir = "out/testout/"
    types_list = ["UndergraduateStudent","TeachingAssistant","Lecturer","Publication","AssistantProfessor",
                  "University","ResearchGroup","Course","FullProfessor","GraduateStudent",
                  "ResearchAssistant","GraduateCourse","Department","AssociateProfessor" ]
    types_list = ["Q11338576"]
    coherence = compute_coherence(types_list, outdir)
    print(coherence)
