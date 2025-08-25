p = output_path / ('{}_spikes_trial_binned.pkl'.format(sess_id_full))
with p.open('wb') as f:
    pkl.dump(event_df, f)
p = output_path / ('{}_licks_trial_binned.pkl'.format(sess_id_full))
with p.open('wb') as f:
    pkl.dump(event_df, f)