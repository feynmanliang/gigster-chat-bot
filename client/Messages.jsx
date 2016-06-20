import React from 'react';

export default class MessageHistory extends React.Component {
  componentDidMount() {
    $('.message-history').scrollTop($('.message-history')[0].scrollHeight); // scroll to bottom
  }
  renderMessages() {
    return this.props.messages.map((message) => {
      return <Message key={message._id} message={message} />
    });
  }
  render() {
    return (
      <div className="message-history">
        {this.renderMessages()}
      </div>
    );
  }
}

MessageHistory.propTypes = {
  messages: React.PropTypes.arrayOf(React.PropTypes.string).isRequired
}

class Message extends React.Component {
  render() {
    return (
      <div className="ui raised segment message">
        <div className="message_header">
          <img className="ui avatar image message_profile-pic" src="/Avatar-blank.jpg"></img>
          <i className="empty star icon message_star"></i>
        </div>
        <span className="message_content">{this.props.message.text}</span>
      </div>
    );
  }
}

Message.propTypes = {
  message: React.PropTypes.object.isRequired
}
