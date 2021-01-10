function [ transformation_matrix2] = get__transformation_matrix(imagePoints,boardsize,grid_size,projection_matrix)

boardsize=boardsize-1;
num_points=size(imagePoints,1);
if boardsize(1)*boardsize(2)~=num_points
    transformation_matrix2=0;
    return;
end

A=zeros(num_points*2,8); B=zeros(num_points*2,1);
for i=1:1:num_points
    index_x= mod((i-1),boardsize(1));
    index_y= floor((i-1)/(boardsize(1)));

    x=index_x*grid_size;
    y=index_y*grid_size;
    
    A(2*i-1,1)=x;A(2*i-1,2)=y;A(2*i-1,3)=1;A(2*i-1,4)=0;
    A(2*i-1,5)=0;A(2*i-1,6)=0;A(2*i-1,7)=-imagePoints(i,1)*x;A(2*i-1,8)=-imagePoints(i,1)*y; 
    B(2*i-1)=imagePoints(i,1);
    A(2*i,1)=0;A(2*i,2)=0;A(2*i,3)=0;A(2*i,4)=x;
    A(2*i,5)=y;A(2*i,6)=1;A(2*i,7)=-imagePoints(i,2)*x;A(2*i,8)=-imagePoints(i,2)*y;     
    B(2*i)=imagePoints(i,2);
end
h1_2_8=A\B;
H=zeros(3,3);
for i=1:1:3
    for j=1:1:3
        if i~=3 || j~=3
           H(i,j)=h1_2_8(3*(i-1)+j);
        else
            H(i,j)=1;
        end
    end
end
transformation_matrix_unnormalized=projection_matrix\H;

normr1=norm(transformation_matrix_unnormalized(:,1));

tn=transformation_matrix_unnormalized/normr1;

r3=cross(tn(:,1),tn(:,2));
%error=tn(:,1)'*tn(:,2);
%fprintf('r1 and r2 error level %f\n', error);
transformation_matrix=[tn(:,1),tn(:,2),r3,tn(:,3)];

u1=tn(:,1);
u2= tn(:,2)- (tn(:,2)'*u1)*u1; u2=u2/norm(u2);
u3=r3-(r3'*u1)*u1- (r3'*u2)*u2; u3=u3/norm(u3);
%M=[tn(:,1),tn(:,2),r3];
U=[u1 u2 u3];
transformation_matrix2=[U,tn(:,3)];
error=norm(transformation_matrix2-transformation_matrix,'fro');
fprintf('error of orthogonal fit %f\n',error);

end

