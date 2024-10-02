y_test_reshaped=y_test.reshape(-1,1)
predicted_y_test_y_test_reshaped=predicted_y_test.reshape(-1,1)

stacked_results=np.hstack((y_test_reshaped,predicted_y_test_y_test_reshaped))

stacked_results_df=pd.DataFrame(stacked_results, columns=["actual","predicted"])
stacked_results_df=stacked_results_df.applymap(lambda x: f"{x:.2e}")
stacked_results_df