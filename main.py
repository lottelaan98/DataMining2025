#!/usr/bin/env python3

from preprocessing import load_op_spam, split_by_fold



def main():
    df = load_op_spam("data/op_spam_v1.4", polarities=("negative",))
    train_df, test_df = split_by_fold(df, train_folds=(1,2,3,4), test_fold=5)

    print(f"Ingelezen: {len(df)} docs | Train: {len(train_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    main()
