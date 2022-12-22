



async function runExample() {
    

  // Create an ONNX inference session with default backend.
  const session = new onnx.InferenceSession();
 
  await session.loadModel("./INP4_CFDtuyereSIO.onnx");

  var x = new Float32Array(1, 4);

  //x = [   38.0000,     0.0000,   300.0000,    27.0000, 308750.3750,    12.1540, 359906.3125,  1459.8170, 24494.0000, 87618.0156]
  

  var x = [];
  x[0] = document.getElementById('box0').value;
  x[1] = document.getElementById('box1').value;
  x[2] = document.getElementById('box2').value;
  x[3] = document.getElementById('box3').value;

 

  
  const tensorX = new onnx.Tensor(x, 'float32', [1, 4]);
  
   
  
  const outputMap = await session.run([tensorX]);
  const outputData = outputMap.get('output1');
  
  
 

  // PREDS DIV 
  const predictions = document.getElementById('predictions');
  
 


  predictions.innerHTML = `<hr> Got an output Tensor of size ${outputData.data.length} with values being: <br/> <table>
  <tr><td> o_tuyere_exit_velo_m_s</td><td> ${outputData.data[0]} </td></tr>
  <tr><td> o_tuyere_t_k</td>          <td> ${outputData.data[1]} </td></tr> 
 </table>   `;
 



}
