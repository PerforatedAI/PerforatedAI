# Google Colab Kernel Startup Troubleshooting

## Issue: Kernel won't start despite selecting L4 GPU

### Quick Fixes (try in order):

1. **Clear browser cache:**
   - Ctrl+Shift+Delete
   - Clear all browsing data
   - Restart browser

2. **Force disconnect:**
   - Runtime → Disconnect and delete runtime
   - Wait 2 minutes
   - Refresh page
   - Runtime → Change runtime type → L4 GPU

3. **Try different browser:**
   - Chrome (recommended)
   - Firefox
   - Edge

4. **Check Colab status:**
   - Go to: https://status.cloud.google.com/
   - Check for Colab outages

5. **Account limits:**
   - You may have hit GPU quota
   - Wait 1-2 hours and try again
   - Or use CPU temporarily

### Alternative: Local Execution Script

If Colab keeps failing, I can create a local Python script that runs the same experiment.