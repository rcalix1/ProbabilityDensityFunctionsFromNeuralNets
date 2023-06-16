
      

async function runExample1() {
    
           ////////////////////////////////////////////////
      
           var x = new Float32Array(1, 6);

           var x = [];
           x[0] = document.getElementById('box0c1').value;
           x[1] = document.getElementById('box1c1').value;
           x[2] = document.getElementById('box2c1').value;
           x[3] = document.getElementById('box3c1').value;
           x[4] = document.getElementById('box4c1').value;
           x[5] = document.getElementById('box5c1').value;
     
           let tensorX = new ort.Tensor('float32', x, [1, 6]);
           let feeds = { float_input: tensorX};
           
           
           
           let session1 = await ort.InferenceSession.create('./xgboost_raceway_coal_burn_perce_ort.onnx');
           let results1 = await session1.run(feeds);
           let outputData1 = results1.variable.data.toFixed(2);
      
         
           let session2 = await ort.InferenceSession.create('./xgboost_raceway_flame_temp_k_ort.onnx');
           let results2 = await session2.run(feeds);
           let outputData2 = results2.variable.data.toFixed(2);
      
      
           let session3 = await ort.InferenceSession.create('./xgboost_raceway_volume_m_ort.onnx');
           let results3 = await session3.run(feeds);
           let outputData3 = results3.variable.data.toFixed(2);
          


  // PREDS DIV 
  let predictions = document.getElementById('predictions1');
  

 predictions.innerHTML = `<hr> Got an output Tensor with values being: <br/> 
 <table>
 
  <tr>
  <td> o_raceway_coal_burn_perce</td>
  <td id="c1td0"> ${outputData1} </td>
  </tr>
  
  <tr>
  <td> o_raceway_flame_temp_k </td>
  <td id="c1td1"> ${outputData2} </td>
  </tr> 
  
  <tr>
  <td> o_raceway_volume_m </td>
  <td id="c1td2"> ${outputData3} </td>
  </tr> 
  
 </table>   `;
 

runDiff();

}


async function runExample2() {
           
           var x = new Float32Array(1, 6);

           var x = [];
           x[0] = document.getElementById('box0c2').value;
           x[1] = document.getElementById('box1c2').value;
           x[2] = document.getElementById('box2c2').value;
           x[3] = document.getElementById('box3c2').value;
           x[4] = document.getElementById('box4c2').value;
           x[5] = document.getElementById('box5c2').value;
           

           let tensorX = new ort.Tensor('float32', x, [1, 6]);
           let feeds = { float_input: tensorX};
           
           
           let session1 = await ort.InferenceSession.create('./xgboost_raceway_coal_burn_perce_ort.onnx');
           let results1 = await session1.run(feeds);
           let outputData1 = results1.variable.data;

 
           let session2 = await ort.InferenceSession.create('./xgboost_raceway_flame_temp_k_ort.onnx');
           let results2 = await session2.run(feeds);
           let outputData2 = results2.variable.data;
      
           let session3 = await ort.InferenceSession.create('./xgboost_raceway_volume_m_ort.onnx');
           let results3 = await session3.run(feeds);
           let outputData3 = results3.variable.data;
          
 
 

  // PREDS DIV 
  let predictions = document.getElementById('predictions2');
  

  predictions.innerHTML = `<hr> Got an output Tensor  with values being: <br/> 
 <table>
 
  <tr>
  <td> o_raceway_coal_burn_perce</td>
  <td id="c2td0"> ${outputData1} </td>
  </tr>
  
  <tr>
  <td> o_raceway_flame_temp_k </td>
  <td id="c2td1"> ${outputData2} </td>
  </tr> 
  
  <tr>
  <td> o_raceway_volume_m </td>
  <td id="c2td2"> ${outputData3} </td>
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
  <td> o_raceway_coal_burn_perce </td>
  <td> ${td0.toFixed(2)} </td>
  </tr>
  
  <tr>
  <td> o_raceway_flame_temp_k </td>
  <td> ${td1.toFixed(2)} </td>
  </tr> 
  
  <tr>
  <td> o_raceway_volume_m  </td>
  <td> ${td2.toFixed(2)} </td>
  </tr> 

  
 </table>   `;
    
}

