import session_analysis
import fileIO


def main():
    """Analyze the data from a single mouse session"""
    plot_path = '../figures'
    mouse = 'HD005'
    date = '2023-08-29'
    sess_ID = mouse + '-' + date

    df = fileIO.load(mouse, date, load_cleaned=True)
    session_analysis.generate_session_raster(df, sess_ID + '_raster', plot_path)
    water = session_analysis.count_water(df)
    bin_counts, session_df, bins = session_analysis.plot_binned_behavior(df,  plot_path, sess_ID + '_bin_plot', plot=False)
    session_analysis.block_analysis(bin_counts, session_df, sess_ID)

    """Analyze data of a mouse across several sessions"""
    mouse = 'MF24'
    sess_list = [(mouse, '2023-07-14'),
                 (mouse, '2023-07-17'),
                 (mouse, '2023-07-18'),
                 (mouse, '2023-07-19')]
    session_analysis.compare_sessions(sess_list, plot_path, plot_name='MF24_sample')


if __name__ == '__main__':
    main()