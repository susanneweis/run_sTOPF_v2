import pandas as pd
import matplotlib.pyplot as plt


def main(path,file, col):

    in_file = f"{path}/{file}.csv"
    out_file = f"{path}/{file}_{col}.png"
    out_file2 = f"{path}/{file}_{col}_2.png"

    df = pd.read_csv(in_file)

    # Make sure top_reg is numeric and sorted
    df["top_reg"] = pd.to_numeric(df["top_reg"])

    plot_df = df.pivot(
        index="movie",
        columns="top_reg",
        values=col
    )

    movie_order = df["movie"].drop_duplicates().tolist()
    plot_df = plot_df.reindex(movie_order)
    plot_df = plot_df[sorted(plot_df.columns)]

    ax = plot_df.plot(kind="bar", figsize=(12, 6))

    ax.set_xlabel("movie")
    ax.set_ylabel(col)
    ax.set_title(f"{col} by movie and top regions")

    plt.xticks(rotation=45, ha="right")
    plt.legend(title="top regions", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()


    # Make sure top_reg is numeric and sorted
    df["top_reg"] = pd.to_numeric(df["top_reg"])

    plot_df = df.pivot(
        index="movie",
        columns="top_reg",
        values=col
    )

    movie_order = df["movie"].drop_duplicates().tolist()
    plot_df = plot_df.reindex(movie_order)
    plot_df = plot_df[sorted(plot_df.columns)]

    # sort region numbers
    plot_df = plot_df[sorted(plot_df.columns)]

    # line plot
    ax = plot_df.plot(
        kind="line",
        marker="o",
        figsize=(12,6)
    )

    ax.set_xlabel("movie")
    ax.set_ylabel(col)
    ax.set_title(f"{col} by movie and top regions")

    ax.set_xticks(range(len(plot_df.index)))
    ax.set_xticklabels(plot_df.index, rotation=45, ha="right")

    plt.legend(title="top regions", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_file2, dpi=300, bbox_inches="tight")
    plt.show()

# Execute script
if __name__ == "__main__":
    main()
