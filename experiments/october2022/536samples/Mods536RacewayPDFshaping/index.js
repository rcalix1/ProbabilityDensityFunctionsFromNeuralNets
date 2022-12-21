



async function runExample1() {
    

  // Create an ONNX inference session with default backend.
  const session = new onnx.InferenceSession();
 
  await session.loadModel("./s536Raceway4by3PDFshapingRC.onnx");

  var x = new Float32Array(1, 4);

  //x = [   38.0000,     0.0000,   300.0000,    27.0000, 308750.3750,    12.1540, 359906.3125,  1459.8170, 24494.0000, 87618.0156]
  

  var x = [];
  x[0] = document.getElementById('box0c1').value;
  x[1] = document.getElementById('box1c1').value;
  x[2] = document.getElementById('box2c1').value;
  x[3] = document.getElementById('box3c1').value;

 

  
  const tensorX = new onnx.Tensor(x, 'float32', [1, 4]);
  
   
  
  const outputMap = await session.run([tensorX]);
  const outputData = outputMap.get('output1');
  
  
 

  // PREDS DIV 
  const predictions = document.getElementById('predictions1');
  
 


  predictions.innerHTML = `<hr> Got an output Tensor of size ${outputData.data.length} with values being: <br/> 
 <table>
  <tr>
  <td> o_raceway_flame_temp_k</td>
  <td id="c1td0"> ${outputData.data[0].toFixed(2)} </td>
  </tr>
  <tr>
  <td> o_raceway_coal_burn_perce</td>
  <td id="c1td1"> ${outputData.data[1].toFixed(2)} </td>
  </tr> 
  <tr>
  <td> o_raceway_volume_m</td>
  <td id="c1td2"> ${outputData.data[2].toFixed(2)} </td>
  </tr>
 </table>   `;
 
runDiff();


}


async function runExample2() {
    

  // Create an ONNX inference session with default backend.
  const session = new onnx.InferenceSession();
 
  await session.loadModel("./s536Raceway4by3PDFshapingRC.onnx");

  var x = new Float32Array(1, 4);

  //x = [   38.0000,     0.0000,   300.0000,    27.0000, 308750.3750,    12.1540, 359906.3125,  1459.8170, 24494.0000, 87618.0156]
  

  var x = [];
  x[0] = document.getElementById('box0c2').value;
  x[1] = document.getElementById('box1c2').value;
  x[2] = document.getElementById('box2c2').value;
  x[3] = document.getElementById('box3c2').value;

 

  
  const tensorX = new onnx.Tensor(x, 'float32', [1, 4]);
  
   
  
  const outputMap = await session.run([tensorX]);
  const outputData = outputMap.get('output1');
  
  
 

  // PREDS DIV 
  const predictions = document.getElementById('predictions2');
  
 


  predictions.innerHTML = `<hr> Got an output Tensor of size ${outputData.data.length} with values being: <br/> 
 <table>
  <tr>
  <td> o_raceway_flame_temp_k</td>
  <td id="c2td0"> ${outputData.data[0].toFixed(2)} </td>
  </tr>
  <tr>
  <td> o_raceway_coal_burn_perce</td>
  <td id="c2td1"> ${outputData.data[1].toFixed(2)} </td>
  </tr> 
  <tr>
  <td> o_raceway_volume_m</td>
  <td id="c2td2"> ${outputData.data[2].toFixed(2)} </td>
  </tr>
 </table>   `;
 
 
runDiff();


}


async function runDiff() {
    
    var c1td0 = parseFloat( document.getElementById('c1td0').innerHTML );
    var c1td1 = parseFloat( document.getElementById('c1td1').innerHTML );
    var c1td2 = parseFloat( document.getElementById('c1td2').innerHTML );
    
    
    var c2td0 = parseFloat( document.getElementById('c2td0').innerHTML );
    var c2td1 = parseFloat( document.getElementById('c2td1').innerHTML );
    var c2td2 = parseFloat( document.getElementById('c2td2').innerHTML );
    
    td0 = c1td0 - c2td0;
    td1 = c1td1 - c2td1;
    td2 = c1td2 - c2td2;

 
     difference.innerHTML = `<hr> Difference is: <br/> 
 <table>
  <tr>
  <td> o_raceway_flame_temp_k</td>
  <td> ${td0.toFixed(2)} </td>
  </tr>
  <tr>
  <td> o_raceway_coal_burn_perce</td>
  <td> ${td1.toFixed(2)} </td>
  </tr> 
  <tr>
  <td> o_raceway_volume_m</td>
  <td> ${td2.toFixed(2)} </td>
  </tr>
 </table>   `;
    
}

