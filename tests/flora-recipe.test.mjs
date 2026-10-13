import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {floraAt,prepareFlora,floraFromRecipe,toMyr} from '../src/flora-field.ts';

export function recipeCases(api){
  const values=[];
  for(let i=0;i<240;i++){
    const d=[Math.cos(i)*Math.sin(i*.3),Math.cos(i*.3),Math.sin(i)*Math.sin(i*.3)];
    const biome={weights:[.1,(i%11)/10,.2,.1],forestPotential:i%3===0,moisture:(i%7)/6,slopeDeg:i%40,ocean:i%19===0};
    const gate={stage:['complex','microbial','sterile'][i%3],availableKinds:i%7===0?[]:i%5===0?['shrub','flower']:undefined};
    for(const t of [-20000,0,12345.6789,20000])values.push(api('FLORA-RECIPE-1',`G001/planet/${i%4}`,d,biome,gate,toMyr(t)));
  }
  return values;
}

test('retained recipes match the previous flora field and exact date replay',()=>{
  const direct=recipeCases(floraAt);
  const prepared=recipeCases((seed,address,d,b,g,t)=>{
    const recipe=prepareFlora(seed,address,d,b,g),before=JSON.stringify(recipe);
    floraFromRecipe(recipe,toMyr(999));
    const value=floraFromRecipe(recipe,t);
    assert.equal(JSON.stringify(recipe),before,'date evaluation never mutates recipe');
    return value;
  });
  assert.deepEqual(prepared,direct);
  // Captured from parent 4c2369d's original floraAt implementation.
  assert.equal(createHash('sha256').update(JSON.stringify(direct)).digest('hex'),'82323efd2b5eb397ad9dc27793138fa4e65c484cdbf360342d6697ee268817e0');
});

test('recipe date boundary still rejects incorrect time units, including absent flora',()=>{
  assert.throws(()=>floraFromRecipe({present:false},NaN),RangeError);
  assert.throws(()=>floraFromRecipe({present:false},Infinity),RangeError);
});
