$json = Get-Content test_fraud_request.json -Raw
$response = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $json
$response | ConvertTo-Json -Depth 10
